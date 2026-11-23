"""Reclamos Sucre: API + páginas (ciudadano y administración) + modelo NLP + alertas por correo."""
import os, io, json, math, time, secrets, smtplib, html, datetime as dt
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from email.message import EmailMessage
from typing import List
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image, ImageOps
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, LargeBinary, ForeignKey, select, func, or_, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

E = os.getenv
BASE = os.path.dirname(__file__)
CATS = ["Recojo de basura", "Agua y alcantarillado", "Alumbrado público", "Baches y vías", "Transporte y tránsito",
        "Seguridad ciudadana", "Ruidos y medio ambiente", "Otros"]
PESO = dict(zip(CATS, [1, 2, 1.5, 1.5, 1, 2.5, 0.5, 0.5]))      # importancia de cada categoría
CELDA = 0.005                                                    # zona = cuadrícula de ~550 m
VENTANA, MIN_MG, MIN_G, COOLDOWN = (int(E("VENTANA_DIAS", 30)), int(E("ALERTA_MIN_RECLAMOS", 3)),
                                    int(E("ALERTA_MIN_GRAVES", 2)), int(E("ALERTA_HORAS", 12)))
AUTOCORR = E("AUTOCORREGIR", "1") == "1"

# ---------------- base de datos ----------------
url = E("DATABASE_URL", "sqlite:///data.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(url, pool_pre_ping=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
Session = sessionmaker(engine); Base = declarative_base()
ahora = lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

class Reclamo(Base):
    __tablename__ = "reclamos"
    id = Column(Integer, primary_key=True); creado = Column(DateTime, default=ahora, index=True)
    texto = Column(Text); cat_ciudadano = Column(String(40)); cat_modelo = Column(String(40)); p_cat = Column(Float)
    veredicto = Column(String(20)); cat_final = Column(String(40), index=True)
    grav = Column(Integer); grav_cont = Column(Float); puntaje = Column(Float)
    lat = Column(Float); lon = Column(Float); fuente_modelo = Column(String(10))
    estado = Column(String(20), default="pendiente"); resuelto_en = Column(DateTime)
class Foto(Base):
    __tablename__ = "fotos"
    id = Column(Integer, primary_key=True); reclamo_id = Column(Integer, ForeignKey("reclamos.id"), index=True); datos = Column(LargeBinary)
class Alerta(Base):
    __tablename__ = "alertas"
    id = Column(Integer, primary_key=True); creada = Column(DateTime, default=ahora)
    cat = Column(String(40)); i = Column(Integer); j = Column(Integer)

def migrar():
    """Agrega las columnas nuevas si la base ya existía (SQLite o PostgreSQL)."""
    cols = {c["name"] for c in inspect(engine).get_columns("reclamos")}
    with engine.begin() as cx:
        if "estado" not in cols: cx.execute(text("ALTER TABLE reclamos ADD COLUMN estado VARCHAR(20) DEFAULT 'pendiente'"))
        if "resuelto_en" not in cols: cx.execute(text("ALTER TABLE reclamos ADD COLUMN resuelto_en TIMESTAMP"))

# ---------------- modelo NLP (notebook 06; versión liviana ONNX del notebook 07, o PyTorch) ----------------
PRED, META, UMBRAL, FUENTE, UMB_G = None, None, 0.8, "reglas", 0.0
NOMBRES_ONNX = ("modelo_onnx", "modelos_onnx")                 # versión liviana (notebook 07): se prueba PRIMERO
NOMBRES_TORCH = ("modelo_reclamos", "modelos_reclamos")        # modelo completo (notebook 06): respaldo
def carpetas_modelo():
    """Carpetas con meta.json, en orden de prioridad: liviano en cualquier ubicación, luego el completo."""
    raices = list(dict.fromkeys([os.getcwd(), BASE, os.path.dirname(BASE)]))
    cands = ([E("MODEL_DIR")] if E("MODEL_DIR") else []) + [os.path.join(r, n) for n in NOMBRES_ONNX + NOMBRES_TORCH for r in raices]
    return [d for d in dict.fromkeys(cands) if os.path.exists(os.path.join(d, "meta.json"))], cands

def cargar_onnx(d, meta):
    import numpy as np, onnxruntime as ort
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json")); tk.enable_truncation(max_length=meta["max_len"])
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.enable_mem_pattern = False; so.enable_cpu_mem_arena = False          # ahorra memoria
    ses = ort.InferenceSession(os.path.join(d, "modelo.int8.onnx"), so, providers=["CPUExecutionProvider"])
    pide_tipos = "token_type_ids" in {i.name for i in ses.get_inputs()}
    sm = lambda x: (np.exp(x - x.max()) / np.exp(x - x.max()).sum()).tolist()
    def predecir(texto):
        e = tk.encode(texto)
        f = {"input_ids": np.array([e.ids], dtype=np.int64), "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        if pide_tipos: f["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
        lc, lg = ses.run(None, f)
        return sm(lc[0]), sm(lg[0])
    return predecir

def cargar_torch(d, meta):
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    tok, bert = AutoTokenizer.from_pretrained(d), AutoModel.from_pretrained(d).eval()
    h = bert.config.hidden_size
    fc_c, fc_g = nn.Linear(h, len(meta["categorias"])), nn.Linear(h, 3)
    cab = torch.load(os.path.join(d, "cabezas.pt"), map_location="cpu")
    fc_c.load_state_dict(cab["cat"]); fc_g.load_state_dict(cab["grav"]); fc_c.eval(); fc_g.eval()
    def predecir(texto):
        with torch.no_grad():
            enc = tok([texto], truncation=True, max_length=meta["max_len"], return_tensors="pt")
            z = bert(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float(); p = (z * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return torch.softmax(fc_c(p), -1)[0].tolist(), torch.softmax(fc_g(p), -1)[0].tolist()
    return predecir

def cargar_modelo():
    if E("USAR_MODELO", "1") != "1": return None, None, "reglas"       # USAR_MODELO=0 fuerza las reglas simples
    dirs, buscadas = carpetas_modelo()
    if not dirs and E("MODEL_REPO"):
        try:
            from huggingface_hub import snapshot_download              # descarga desde Hugging Face (gratis)
            dirs = [snapshot_download(E("MODEL_REPO"), token=E("HF_TOKEN"))]
        except Exception as e: print("No se pudo descargar MODEL_REPO:", repr(e))
    for d in dirs:                                                     # prueba en orden; si uno falla, sigue con el siguiente
        try:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            if meta.get("formato") == "onnx-int8": f, fuente = cargar_onnx(d, meta), "onnx"
            else: f, fuente = cargar_torch(d, meta), "bert"
            print(f"Modelo cargado ({fuente}) desde: {d}"); return f, meta, fuente
        except Exception as e:
            print(f"No se pudo cargar {d}: {e!r}. Se prueba la siguiente opción.")
    print("MODELO NO DISPONIBLE, se usan reglas simples. Carpetas revisadas:", " | ".join(buscadas)); return None, None, "reglas"

ALTA = ["peligro", "herido", "accidente", "incendio", "gas", "cable", "chispa", "inund", "derrumbe", "ninos", "anciano", "urgente", "hundimiento", "hospital", "arma", "cuchillo"]
MEDIA = ["dias", "semanas", "meses", "sin agua", "sin luz", "olor", "robo", "asalto", "oscuro", "rata"]
def reglas(t):     # respaldo mientras no haya modelo entrenado
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")
    g = min(1, sum(.35 for k in ALTA if k in t) + sum(.15 for k in MEDIA if k in t))
    return [.8, .15, .05] if g < .15 else [.2, .6, .2] if g < .5 else [.05, .25, .7]

def analizar(texto, ciud):
    if PRED:
        pc, pg = PRED(texto); cats = META["categorias"]
        k = max(range(len(pc)), key=pc.__getitem__); conf, mod, fuente = pc[k], cats[k], FUENTE
        v = "coincide" if mod == ciud else ("posible error" if conf >= UMBRAL else "dudosa")
        final = mod if (v == "posible error" and AUTOCORR) else ciud
    else:
        pg, mod, conf, v, final, fuente = reglas(texto), ciud, 1.0, "coincide", ciud, "reglas"
    gc = 0.5 * pg[1] + pg[2]
    g = max(range(3), key=pg.__getitem__)
    if UMB_G > 0 and pg[2] >= UMB_G: g = 2          # prioriza no pasar por alto un caso grave
    return dict(cat_modelo=mod, p_cat=conf, veredicto=v, cat_final=final, grav=g,
                grav_cont=gc, puntaje=PESO[final] * 1.2 + gc * 5, fuente_modelo=fuente)

# ---------------- zonas y alertas ----------------
celda = lambda la, lo: (math.floor(la / CELDA), math.floor(lo / CELDA))
emergencia = lambda n_mg, n_g: n_mg >= MIN_MG or n_g >= MIN_G
def nivel(score, n_mg, n_g):
    return "rojo" if (emergencia(n_mg, n_g) or score >= 14) else "amarillo" if score >= 7 else "verde"
def zonas_de(rows):
    z = {}
    for r in rows:
        d = z.setdefault(celda(r.lat, r.lon), dict(score=0.0, n=0, n_mg=0, n_g=0))
        d["score"] += r.puntaje; d["n"] += 1; d["n_mg"] += r.grav >= 1; d["n_g"] += r.grav == 2
    return z
def en_ventana(db, cat=None):
    q = select(Reclamo).where(Reclamo.creado >= ahora() - dt.timedelta(days=VENTANA), or_(Reclamo.estado.is_(None), Reclamo.estado != "solucionado"))
    return list(db.scalars(q.where(Reclamo.cat_final == cat) if cat else q))

def enviar_correo(asunto, cuerpo):
    para = E("ALERT_EMAIL")
    if not para: print("ALERTA (sin ALERT_EMAIL):", asunto); return
    try:
        if E("RESEND_API_KEY"):
            r = httpx.post("https://api.resend.com/emails", timeout=20, headers={"Authorization": "Bearer " + E("RESEND_API_KEY")},
                           json={"from": E("MAIL_FROM", "Alertas <onboarding@resend.dev>"), "to": [para], "subject": asunto, "html": cuerpo})
            print("Resend:", r.status_code, r.text[:200])
        elif E("SMTP_HOST"):
            m = EmailMessage(); m["Subject"], m["From"], m["To"] = asunto, E("SMTP_USER"), para
            m.set_content("Abre este correo en un lector con HTML."); m.add_alternative(cuerpo, subtype="html")
            with smtplib.SMTP(E("SMTP_HOST"), int(E("SMTP_PORT", 587)), timeout=20) as s:
                s.starttls(); s.login(E("SMTP_USER"), E("SMTP_PASSWORD")); s.send_message(m)
        else: print("ALERTA (sin servicio de correo):", asunto)
    except Exception as e: print("Error enviando correo:", repr(e))

# ---------- resumen automático de una zona (sin entrenamiento: estadística + extracción de frases) ----------
import re, unicodedata
BOL = dt.timezone(dt.timedelta(hours=-4))                         # hora de Bolivia (sin horario de verano)
sin_tilde = lambda w: "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")
PARADAS = set(sin_tilde(w) for w in """para como pero este esta esto estos estas otro otra otros otras porque cuando donde desde hasta sobre entre
    tambien tiene tienen estan estar hace hacer hecho cada tres dos ayer anoche hoy ahora noche tarde manana dias semana semanas meses anos
    todo todos toda todas mucho muchos muy solo cual cuales quien algo nada ellos ellas nosotros vecinos zona calle cerca junto frente
    hubo han son fue era sido ser sus nuestro nuestra nuestros nuestras alli aqui esas esos ese esa quiero quisiera favor gracias""".split())
def palabras_clave(textos, n=5):
    """Palabras que aparecen en más reclamos (cada reclamo cuenta una vez por palabra)."""
    cnt, forma = {}, {}
    for t in textos:
        for w in set(re.findall(r"[a-záéíóúñü]{4,}", t.lower())):
            k = sin_tilde(w)
            if k not in PARADAS: cnt[k] = cnt.get(k, 0) + 1; forma.setdefault(k, w)
    top = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:n]
    return [(forma[k], c) for k, c in top if c >= 2] or [(forma[k], c) for k, c in top[:3]]
def mas_representativo(zona):
    """El reclamo cuyas palabras son las más compartidas con los demás de la zona."""
    df = dict((sin_tilde(w), c) for w, c in palabras_clave([x.texto for x in zona], n=50))
    def puntaje(x):
        ws = {sin_tilde(w) for w in re.findall(r"[a-záéíóúñü]{4,}", x.texto.lower())} - PARADAS
        return sum(df.get(w, 0) for w in ws) / (len(ws) + 3)
    return max(zona, key=puntaje)
hora_local = lambda d: d.replace(tzinfo=dt.timezone.utc).astimezone(BOL).strftime("%d/%m %H:%M")
def resumen_zona(zona, fot):
    """Devuelve (resumen en una frase, palabras clave, reclamos más graves, reclamo representativo)."""
    n = len(zona); g = sum(x.grav == 2 for x in zona); m = sum(x.grav == 1 for x in zona)
    t0, t1 = min(x.creado for x in zona), max(x.creado for x in zona)
    temas = palabras_clave([x.texto for x in zona])
    frase = f"{n} reclamos en la misma zona ({g} graves, {m} moderados, {n - g - m} leves), registrados entre {hora_local(t0)} y {hora_local(t1)} (hora de Bolivia)."
    if temas: frase += " Temas más mencionados: " + ", ".join(f"{w} ({c})" for w, c in temas) + "."
    if fot: frase += f" {len(fot)} reclamos incluyen fotos."
    dud = sum(x.veredicto in ("dudosa", "posible error") for x in zona)
    if dud: frase += f" {dud} tienen la categoría en duda o corregida por el modelo."
    graves = sorted(zona, key=lambda x: (-x.grav, -(x.grav_cont or 0), -x.puntaje))[:3]
    return frase, temas, graves, mas_representativo(zona)

def armar_correo(cat, zona, fot, la, lo):
    frase, temas, graves, rep = resumen_zona(zona, fot)
    e = html.escape
    item = lambda x: f"<li><b>{['Leve', 'Moderada', 'Grave'][x.grav]}</b> · {hora_local(x.creado)}{' · 📷' if x.id in fot else ''}<br>{e(x.texto[:200])}</li>"
    return (f"<h2>⚠️ Posible emergencia: {e(cat)}</h2><p><b>Resumen:</b> {e(frase)}</p><p><b>Por favor revísalo.</b></p>"
            f"<h3>Reclamos más graves</h3><ol>{''.join(item(x) for x in graves)}</ol>"
            f"<h3>Reclamo más representativo</h3><ul>{item(rep)}</ul>"
            f"<p><a href='https://www.google.com/maps?q={la:.5f},{lo:.5f}'>Ver la zona en Google Maps</a> · "
            f"<a href='{E('PUBLIC_URL', '')}/admin'>Abrir el panel de administración</a></p>")

def revisar_alerta(db, r, bg):
    i, j = celda(r.lat, r.lon)
    zona = [x for x in en_ventana(db, r.cat_final) if celda(x.lat, x.lon) == (i, j)]
    mg, g = [x for x in zona if x.grav >= 1], [x for x in zona if x.grav == 2]
    if not emergencia(len(mg), len(g)): return
    if db.scalars(select(Alerta).where(Alerta.cat == r.cat_final, Alerta.i == i, Alerta.j == j,
                  Alerta.creada >= ahora() - dt.timedelta(hours=COOLDOWN))).first(): return
    db.add(Alerta(cat=r.cat_final, i=i, j=j)); db.commit()
    la, lo = (i + .5) * CELDA, (j + .5) * CELDA
    cuerpo = armar_correo(r.cat_final, zona, fotos_de(db, [x.id for x in zona]), la, lo)
    bg.add_task(enviar_correo, f"⚠️ Emergencia en {r.cat_final}: {len(zona)} reclamos ({len(g)} graves)", cuerpo)

# ---------------- seguridad ----------------
_h = defaultdict(deque)
def limitar(clave, n, seg):
    q, t = _h[clave], time.time()
    while q and q[0] < t - seg: q.popleft()
    if len(q) >= n: raise HTTPException(429, "Demasiados intentos. Inténtalo más tarde.")
    q.append(t)
ip_de = lambda req: (req.headers.get("x-forwarded-for", "").split(",")[0].strip() or req.client.host)
def exigir_admin(req: Request):
    if not req.session.get("admin"): raise HTTPException(401, "No autorizado")

@asynccontextmanager
async def vida(app):
    global PRED, META, UMBRAL, FUENTE, UMB_G
    Base.metadata.create_all(engine); migrar(); PRED, META, FUENTE = cargar_modelo()
    UMBRAL = (META or {}).get("umbral_categoria", 0.8)
    UMB_G = float(E("UMBRAL_GRAVE") or (META or {}).get("umbral_grave") or 0); yield   # 0 = clase más probable

app = FastAPI(lifespan=vida, docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=E("SECRET_KEY") or secrets.token_hex(32), same_site="lax",
                   https_only=E("HTTPS_ONLY", "1") == "1", max_age=8 * 3600)

@app.get("/api/config")
def config(): return {"carto_key": E("CARTO_KEY", ""), "categorias": CATS, "modelo": FUENTE}

@app.post("/api/reclamos")
async def crear(req: Request, bg: BackgroundTasks, texto: str = Form(...), categoria: str = Form(...), lat: float = Form(...),
                lon: float = Form(...), web: str = Form(""), fotos: List[UploadFile] = File(default=[])):
    if web: return {"ok": True}                                   # trampa para robots
    limitar("rec:" + ip_de(req), 10, 3600)
    texto = texto.strip()
    if not 10 <= len(texto) <= 1500: raise HTTPException(400, "El reclamo debe tener entre 10 y 1500 caracteres.")
    if categoria not in CATS: raise HTTPException(400, "Categoría inválida.")
    if not (-19.3 <= lat <= -18.8 and -65.5 <= lon <= -65.0): raise HTTPException(400, "La ubicación debe estar en Sucre.")
    imgs = []
    for f in fotos[:3]:
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(await f.read(8_000_000)))).convert("RGB")   # quita datos EXIF (GPS)
            im.thumbnail((1024, 1024)); b = io.BytesIO(); im.save(b, "JPEG", quality=80); imgs.append(b.getvalue())
        except Exception: raise HTTPException(400, "Una de las fotos no es una imagen válida.")
    with Session() as db:
        r = Reclamo(texto=texto, cat_ciudadano=categoria, lat=round(lat, 5), lon=round(lon, 5), **analizar(texto, categoria))
        db.add(r); db.flush(); db.add_all([Foto(reclamo_id=r.id, datos=b) for b in imgs]); db.commit()
        revisar_alerta(db, r, bg); return {"ok": True, "id": r.id}

class Login(BaseModel): user: str; password: str
@app.post("/api/login")
def login(d: Login, req: Request):
    if not E("ADMIN_PASSWORD"): raise HTTPException(503, "Falta configurar ADMIN_PASSWORD")
    limitar("login:" + ip_de(req), 5, 900)
    ok = secrets.compare_digest(d.user, E("ADMIN_USER", "admin")) & secrets.compare_digest(d.password, E("ADMIN_PASSWORD"))
    if not ok: raise HTTPException(401, "Usuario o contraseña incorrectos")
    req.session["admin"] = True; return {"ok": True}
@app.post("/api/logout")
def logout(req: Request): req.session.clear(); return {"ok": True}
@app.get("/api/yo")
def yo(req: Request): return {"admin": bool(req.session.get("admin"))}

@app.get("/api/admin/datos")
def datos(req: Request, categoria: str):
    exigir_admin(req)
    with Session() as db:
        todo = en_ventana(db); resumen = {}
        for c in CATS:
            filas = [r for r in todo if r.cat_final == c]
            resumen[c] = dict(n=len(filas), rojas=sum(nivel(z["score"], z["n_mg"], z["n_g"]) == "rojo" for z in zonas_de(filas).values()))
        filas = [r for r in todo if r.cat_final == categoria]
        fot = {}
        for fid, rid in db.execute(select(Foto.id, Foto.reclamo_id).where(Foto.reclamo_id.in_([r.id for r in filas] or [0]))):
            fot.setdefault(rid, []).append(fid)
        zs = [dict(i=i, j=j, lat0=i * CELDA, lon0=j * CELDA, nivel=nivel(z["score"], z["n_mg"], z["n_g"]), **z) for (i, j), z in zonas_de(filas).items()]
        rs = [dict(id=r.id, texto=r.texto, lat=r.lat, lon=r.lon, grav=r.grav, grav_cont=round(r.grav_cont, 2), puntaje=round(r.puntaje, 1),
                   fecha=r.creado.isoformat(), fotos=fot.get(r.id, []), veredicto=r.veredicto, cat_ciudadano=r.cat_ciudadano,
                   cat_modelo=r.cat_modelo, p_cat=round(r.p_cat or 0, 2), celda=list(celda(r.lat, r.lon))) for r in filas]
        n_sol = db.scalar(select(func.count()).select_from(Reclamo).where(Reclamo.estado == "solucionado"))
        return dict(n_sol=n_sol, resumen=resumen, zonas=zs, reclamos=sorted(rs, key=lambda x: -x["puntaje"]), celda=CELDA)

def fotos_de(db, ids):
    fot = {}
    for fid, rid in db.execute(select(Foto.id, Foto.reclamo_id).where(Foto.reclamo_id.in_(ids or [0]))):
        fot.setdefault(rid, []).append(fid)
    return fot

class ZonaIn(BaseModel): categoria: str; i: int; j: int
@app.post("/api/admin/zona/resolver")
def resolver_zona(d: ZonaIn, req: Request):
    """Marca como solucionados todos los reclamos pendientes de esa categoría y zona."""
    exigir_admin(req)
    with Session() as db:
        filas = [r for r in db.scalars(select(Reclamo).where(Reclamo.cat_final == d.categoria))
                 if (r.estado or "pendiente") != "solucionado" and celda(r.lat, r.lon) == (d.i, d.j)]
        for r in filas: r.estado, r.resuelto_en = "solucionado", ahora()
        for a in db.scalars(select(Alerta).where(Alerta.cat == d.categoria, Alerta.i == d.i, Alerta.j == d.j)): db.delete(a)   # una nueva emergencia vuelve a avisar
        db.commit(); return {"ok": True, "resueltos": len(filas)}

class EstadoIn(BaseModel): estado: str
@app.post("/api/admin/reclamo/{rid}/estado")
def estado_reclamo(rid: int, d: EstadoIn, req: Request):
    exigir_admin(req)
    if d.estado not in ("solucionado", "pendiente"): raise HTTPException(400, "Estado inválido")
    with Session() as db:
        r = db.get(Reclamo, rid)
        if not r: raise HTTPException(404)
        r.estado = d.estado; r.resuelto_en = ahora() if d.estado == "solucionado" else None; db.commit(); return {"ok": True}

@app.get("/api/admin/solucionados")
def solucionados(req: Request):
    exigir_admin(req)
    with Session() as db:
        filas = list(db.scalars(select(Reclamo).where(Reclamo.estado == "solucionado").order_by(Reclamo.resuelto_en.desc()).limit(300)))
        fot = fotos_de(db, [r.id for r in filas])
        return [dict(id=r.id, texto=r.texto, cat=r.cat_final, cat_ciudadano=r.cat_ciudadano, grav=r.grav, fecha=r.creado.isoformat(),
                     resuelto=(r.resuelto_en or r.creado).isoformat(), fotos=fot.get(r.id, [])) for r in filas]

@app.get("/api/fotos/{fid}")
def foto(fid: int, req: Request):
    exigir_admin(req)
    with Session() as db:
        f = db.get(Foto, fid)
        if not f: raise HTTPException(404)
        return Response(f.datos, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

ST = os.path.join(BASE, "static")
@app.get("/admin")
def admin_page(): return FileResponse(os.path.join(ST, "admin.html"))
app.mount("/", StaticFiles(directory=ST, html=True), name="static")
