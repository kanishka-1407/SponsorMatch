"""SQLite backed connection requests for SponsorMatch."""
from datetime import datetime, timedelta, timezone
import hashlib
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, HttpUrl

DB_PATH = Path(os.getenv("SPONSORMATCH_DB", Path(__file__).with_name("sponsormatch.sqlite3")))
router = APIRouter(prefix="/pitch", tags=["pitch requests"])

def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('startup','investor')));
        CREATE TABLE IF NOT EXISTS startup_profiles(user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, company TEXT NOT NULL, sector TEXT, stage TEXT, city TEXT, summary TEXT, contact_email TEXT);
        CREATE TABLE IF NOT EXISTS investor_profiles(user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, name TEXT NOT NULL, sectors TEXT, stages TEXT, cities TEXT, summary TEXT, contact_email TEXT);
        CREATE TABLE IF NOT EXISTS pitch_requests(id INTEGER PRIMARY KEY, sender_id INTEGER NOT NULL REFERENCES users(id), recipient_id INTEGER NOT NULL REFERENCES users(id), message TEXT NOT NULL CHECK(length(message)<=300), deck_url TEXT, match_score REAL CHECK(match_score IS NULL OR match_score BETWEEN 0 AND 1), match_reasons TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, info_request TEXT);
        CREATE INDEX IF NOT EXISTS pitch_pair_status ON pitch_requests(sender_id,recipient_id,status);
        CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), request_id INTEGER REFERENCES pitch_requests(id), message TEXT NOT NULL, created_at TEXT NOT NULL, read INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS tokens(token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL);
        ''')
init_db()

def stamp(): return datetime.now(timezone.utc).isoformat()
def fail(code, msg): raise HTTPException(code, msg)

class Register(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=128)
    role: str
    company: Optional[str] = Field(default=None, max_length=120)
    name: Optional[str] = Field(default=None, max_length=120)
    sector: Optional[str] = Field(default=None, max_length=80)
    stage: Optional[str] = Field(default=None, max_length=60)
    city: Optional[str] = Field(default=None, max_length=80)
    summary: Optional[str] = Field(default=None, max_length=1000)

class Login(BaseModel): email: str; password: str
class PitchDraft(BaseModel): recipient_id: int
class SendPitch(BaseModel):
    recipient_id: int
    message: str = Field(min_length=1, max_length=300)
    deck_url: Optional[HttpUrl] = None
    match_score: Optional[float] = Field(default=None, ge=0, le=1)
    match_reasons: list[str] = Field(default_factory=list, max_length=20)
class ResponseAction(BaseModel):
    action: str
    message: Optional[str] = Field(default=None, max_length=1000)

def current_user(authorization: str = Header(default="")):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token: fail(401,"Bearer token required")
    with db() as c:
        row=c.execute("SELECT u.* FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token_hash=? AND t.expires_at>?",(hashlib.sha256(token.encode()).hexdigest(),stamp())).fetchone()
    if not row: fail(401,"Invalid or expired token")
    return row

@router.post('/register', status_code=201)
def register(body: Register):
    if body.role not in ('startup','investor'): fail(422,'role must be startup or investor')
    if body.role=='startup' and not body.company: fail(422,'company is required for startup')
    if body.role=='investor' and not body.name: fail(422,'name is required for investor')
    with db() as c:
        try:
            cur=c.execute('INSERT INTO users(email,password_hash,role) VALUES(?,?,?)',(body.email.lower(),hashlib.scrypt(body.password.encode(),salt=(salt:=secrets.token_bytes(16)),n=2**14,r=8,p=1).hex()+':'+salt.hex(),body.role))
        except sqlite3.IntegrityError: fail(409,'Email already registered')
        uid=cur.lastrowid
        if body.role=='startup': c.execute('INSERT INTO startup_profiles VALUES(?,?,?,?,?,?,?)',(uid,body.company,body.sector,body.stage,body.city,body.summary,body.email.lower()))
        else: c.execute('INSERT INTO investor_profiles VALUES(?,?,?,?,?,?,?)',(uid,body.name,body.sector,body.stage,body.city,body.summary,body.email.lower()))
    return {'id':uid,'email':body.email.lower(),'role':body.role}

@router.post('/login')
def login(body: Login):
    with db() as c: row=c.execute('SELECT * FROM users WHERE email=?',(body.email.lower(),)).fetchone()
    if not row: fail(401,'Invalid credentials')
    stored,salt=row['password_hash'].split(':')
    if not secrets.compare_digest(hashlib.scrypt(body.password.encode(),salt=bytes.fromhex(salt),n=2**14,r=8,p=1).hex(),stored): fail(401,'Invalid credentials')
    token=secrets.token_urlsafe(32); expiry=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    with db() as c: c.execute('INSERT INTO tokens VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),row['id'],expiry))
    return {'access_token':token,'token_type':'bearer','expires_at':expiry}

def target_profile(c, user):
    if user['role']=='startup': return c.execute('SELECT * FROM startup_profiles WHERE user_id=?',(user['id'],)).fetchone()
    return c.execute('SELECT * FROM investor_profiles WHERE user_id=?',(user['id'],)).fetchone()

@router.post('/generate')
def generate(body: PitchDraft, user=Depends(current_user)):
    with db() as c:
        target=c.execute('SELECT * FROM users WHERE id=?',(body.recipient_id,)).fetchone()
        if not target or target['role']==user['role']: fail(404,'Opposite-role recipient not found')
        own=target_profile(c,user)
        other=(c.execute('SELECT * FROM investor_profiles WHERE user_id=?',(target['id'],)).fetchone() if target['role']=='investor' else c.execute('SELECT * FROM startup_profiles WHERE user_id=?',(target['id'],)).fetchone())
    if user['role']=='startup':
        company=own['company']; name=other['name']; sector=own['sector']
        msg=f"Hi {name}, {company} is building in {sector or 'our sector'}. Your investment focus looks aligned; could we share our plans?" if sector else f"Hi {name}, {company} is building an ambitious startup. Your investment focus looks aligned; could we share our plans?"
        reasons=[f"Startup sector: {sector}"] if sector else ['Profile-based introduction']
    else:
        company=other['company']; name=own['name']; sector=other['sector']
        msg=f"Hi {company}, I’m {name}. I invest in {sector or 'early-stage companies'} and would like to learn more about your startup."; reasons=[f"Investor interest: {sector}"] if sector else ['Profile-based introduction']
    score=None
    # Reuse the repository's trained startup -> investor ranker and its profile data.
    if user['role']=='startup':
        try:
            import api
            investor_name=other['name']
            request=api.StartupRequest(sector_group=own['sector'] if own['sector'] in api.SECTORS else 'Other',
                stage_bucket=own['stage'] if own['stage'] in api.STAGES else 'Undisclosed',
                city_group=own['city'] if own['city'] in api.CITIES else 'Unknown',
                amount_bin='Undisclosed',company_age=0)
            ranked=api.get_ranker(request).scores(__import__('pandas').Series(request.model_dump(exclude={'top_k'})))
            score=next((max(0.0,min(1.0,float(s))) for n,s in ranked if str(n)==investor_name),None)
            if score is not None:
                profile=api.read_csv_or_empty(api.INVESTOR_PROFILES_PATH)
                detail=api.profile_details(investor_name,profile)
                sectors=str(detail.get('top_sectors') or '')
                if own['sector'] and own['sector'] in sectors: reasons.append(f"Investor's portfolio includes {own['sector']}")
                if own['stage'] and own['stage'] in str(detail.get('top_stages') or ''): reasons.append(f"Investor has backed {own['stage']} companies")
                if not reasons: reasons.append('Score from trained investor ranking model')
        except Exception:
            score=None
    return {'message':msg[:300],'match_score':score,'match_reasons':reasons,'editable':True}

@router.post('/requests',status_code=201)
def send(body: SendPitch,user=Depends(current_user)):
    now=datetime.now(timezone.utc); cutoff=(now-timedelta(days=7)).isoformat(); ts=now.isoformat()
    with db() as c:
        target=c.execute('SELECT * FROM users WHERE id=?',(body.recipient_id,)).fetchone()
        if not target or target['role']==user['role']: fail(404,'Opposite-role recipient not found')
        pair=(user['id'],target['id']) if user['role']=='startup' else (target['id'],user['id'])
        if c.execute("SELECT 1 FROM pitch_requests WHERE sender_id=? AND recipient_id=? AND status IN ('SENT','VIEWED','INFO_REQUESTED')",pair).fetchone(): fail(409,'A pending request already exists for this pair')
        if c.execute("SELECT count(*) n FROM pitch_requests WHERE sender_id=? AND created_at>=?",(user['id'],cutoff)).fetchone()['n']>=5: fail(429,'Limit is five outgoing requests per rolling seven days')
        score=body.match_score
        reasons=body.match_reasons
        if score is None:
            score=None
            if user['role']=='startup':
                # Derive the score from the same real match context as draft generation.
                draft=generate(PitchDraft(recipient_id=body.recipient_id),user)
                score=draft['match_score']; reasons=reasons or draft['match_reasons']
            reasons=reasons or ['No trained ranking score available for this profile pair']
        cur=c.execute('INSERT INTO pitch_requests(sender_id,recipient_id,message,deck_url,match_score,match_reasons,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',(user['id'],target['id'],body.message,str(body.deck_url) if body.deck_url else None,score,' | '.join(reasons),'SENT',ts,ts))
        rid=cur.lastrowid; c.execute('INSERT INTO notifications(user_id,request_id,message,created_at) VALUES(?,?,?,?)',(target['id'],rid,'New pitch request',ts))
    return {'id':rid,'status':'SENT','match_score':score,'match_reasons':reasons}

def serialize(c,row,viewer):
    accepted=row['status']=='ACCEPTED'; peer=row['sender_id'] if viewer['id']==row['recipient_id'] else row['recipient_id']
    profile=c.execute('SELECT role,email FROM users WHERE id=?',(peer,)).fetchone()
    return {'id':row['id'],'sender_id':row['sender_id'],'recipient_id':row['recipient_id'],'message':row['message'],'deck_url':row['deck_url'],'match_score':row['match_score'],'match_reasons':row['match_reasons'].split(' | '),'status':row['status'],'created_at':row['created_at'],'contact_email':profile['email'] if accepted else None}

@router.get('/requests')
def list_requests(user=Depends(current_user)):
    with db() as c:
        rows=c.execute('SELECT * FROM pitch_requests WHERE sender_id=? OR recipient_id=? ORDER BY created_at DESC',(user['id'],user['id'])).fetchall()
        out=[]
        for row in rows:
            if row['recipient_id']==user['id'] and row['status']=='SENT':
                c.execute("UPDATE pitch_requests SET status='VIEWED',updated_at=? WHERE id=?",(stamp(),row['id']))
                row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(row['id'],)).fetchone()
            out.append(serialize(c,row,user))
    return out

@router.get('/inbox')
def inbox(user=Depends(current_user)):
    if user['role']!='investor': fail(403,'Investor role required')
    with db() as c: rows=c.execute('SELECT * FROM pitch_requests WHERE recipient_id=? ORDER BY created_at DESC',(user['id'],)).fetchall(); return [serialize(c,r,user) for r in rows]

@router.get('/requests/{rid}')
def view_request(rid:int,user=Depends(current_user)):
    with db() as c:
        row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(rid,)).fetchone()
        if not row or user['id'] not in (row['sender_id'],row['recipient_id']): fail(404,'Request not found')
        if row['recipient_id']==user['id'] and row['status']=='SENT': c.execute("UPDATE pitch_requests SET status='VIEWED',updated_at=? WHERE id=?",(stamp(),rid)); row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(rid,)).fetchone()
        return serialize(c,row,user)

@router.post('/requests/{rid}/respond')
def respond(rid:int,body:ResponseAction,user=Depends(current_user)):
    if body.action not in ('accept','decline','request_info'): fail(422,'action must be accept, decline, or request_info')
    with db() as c:
        row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(rid,)).fetchone()
        if not row or row['recipient_id']!=user['id']: fail(404,'Request not found')
        if row['status'] not in ('SENT','VIEWED','INFO_REQUESTED'): fail(409,'Request is no longer actionable')
        status={'accept':'ACCEPTED','decline':'DECLINED','request_info':'INFO_REQUESTED'}[body.action]
        c.execute('UPDATE pitch_requests SET status=?,info_request=?,updated_at=? WHERE id=?',(status,body.message if body.action=='request_info' else None,stamp(),rid))
        c.execute('INSERT INTO notifications(user_id,request_id,message,created_at) VALUES(?,?,?,?)',(row['sender_id'],rid,{'accept':'Request accepted','decline':'Request declined','request_info':'More information requested'}[body.action],stamp()))
        row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(rid,)).fetchone(); return serialize(c,row,user)

@router.post('/requests/{rid}/withdraw')
def withdraw(rid:int,user=Depends(current_user)):
    with db() as c:
        row=c.execute('SELECT * FROM pitch_requests WHERE id=?',(rid,)).fetchone()
        if not row or row['sender_id']!=user['id']: fail(404,'Request not found')
        if row['status'] not in ('SENT','VIEWED','INFO_REQUESTED'): fail(409,'Request can no longer be withdrawn')
        c.execute("UPDATE pitch_requests SET status='WITHDRAWN',updated_at=? WHERE id=?",(stamp(),rid)); return {'id':rid,'status':'WITHDRAWN'}

@router.get('/notifications')
def notifications(user=Depends(current_user)):
    with db() as c: return [dict(r) for r in c.execute('SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC',(user['id'],)).fetchall()]

@router.post('/notifications/{nid}/read')
def mark_read(nid:int,user=Depends(current_user)):
    with db() as c:
        cur=c.execute('UPDATE notifications SET read=1 WHERE id=? AND user_id=?',(nid,user['id']))
        if not cur.rowcount: fail(404,'Notification not found')
    return {'id':nid,'read':True}
