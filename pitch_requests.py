"""SQLite backed connection requests for SponsorMatch."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, HttpUrl, EmailStr, field_validator, model_validator

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
        CREATE TABLE IF NOT EXISTS startup_profiles(user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, company TEXT NOT NULL, sector TEXT, stage TEXT, city TEXT, summary TEXT, contact_email TEXT, amount_usd REAL, company_age REAL NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS investor_profiles(user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, name TEXT NOT NULL, sectors TEXT, stages TEXT, cities TEXT, summary TEXT, contact_email TEXT, min_ticket_usd REAL, max_ticket_usd REAL);
        CREATE TABLE IF NOT EXISTS pitch_requests(id INTEGER PRIMARY KEY, sender_id INTEGER NOT NULL REFERENCES users(id), recipient_id INTEGER NOT NULL REFERENCES users(id), message TEXT NOT NULL CHECK(length(message)<=300), deck_url TEXT, match_score REAL CHECK(match_score IS NULL OR match_score BETWEEN 0 AND 1), match_reasons TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, info_request TEXT);
        CREATE INDEX IF NOT EXISTS pitch_pair_status ON pitch_requests(sender_id,recipient_id,status);
        CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), request_id INTEGER REFERENCES pitch_requests(id), message TEXT NOT NULL, created_at TEXT NOT NULL, read INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS tokens(token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS login_attempts(email TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0, blocked_until TEXT);
        CREATE TABLE IF NOT EXISTS investor_saved_startups(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, company_name TEXT NOT NULL, company_domain TEXT, details_json TEXT NOT NULL, saved_at TEXT NOT NULL, UNIQUE(user_id, company_name));
        ''')
        existing = {row['name'] for row in c.execute('PRAGMA table_info(startup_profiles)')}
        if 'amount_usd' not in existing:
            c.execute('ALTER TABLE startup_profiles ADD COLUMN amount_usd REAL')
        if 'company_age' not in existing:
            c.execute('ALTER TABLE startup_profiles ADD COLUMN company_age REAL NOT NULL DEFAULT 0')
        investor_columns = {row['name'] for row in c.execute('PRAGMA table_info(investor_profiles)')}
        if 'min_ticket_usd' not in investor_columns:
            c.execute('ALTER TABLE investor_profiles ADD COLUMN min_ticket_usd REAL')
        if 'max_ticket_usd' not in investor_columns:
            c.execute('ALTER TABLE investor_profiles ADD COLUMN max_ticket_usd REAL')
init_db()

def stamp(): return datetime.now(timezone.utc).isoformat()
def fail(code, msg): raise HTTPException(code, msg)

class Register(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    role: str
    company: Optional[str] = Field(default=None, max_length=120)
    name: Optional[str] = Field(default=None, max_length=120)
    sector: Optional[str] = None
    stage: Optional[str] = None
    city: Optional[str] = None
    summary: Optional[str] = Field(default=None, max_length=1000)
    amount_usd: Optional[float] = Field(default=None, ge=0)
    company_age: float = Field(default=0, ge=0, le=200)
    preferred_sectors: list[str] = Field(default_factory=list, max_length=14)
    preferred_stages: list[str] = Field(default_factory=list, max_length=7)
    preferred_cities: list[str] = Field(default_factory=list, max_length=11)
    min_ticket_usd: Optional[float] = Field(default=None, ge=0)
    max_ticket_usd: Optional[float] = Field(default=None, ge=0)

    @field_validator("sector")
    @classmethod
    def allowed_sector(cls, value):
        from api import SECTORS
        if value is not None and value not in SECTORS:
            raise ValueError("Choose a sector from /options")
        return value

    @field_validator("stage")
    @classmethod
    def allowed_stage(cls, value):
        from api import STAGES
        if value is not None and value not in STAGES:
            raise ValueError("Choose a stage from /options")
        return value

    @field_validator("city")
    @classmethod
    def allowed_city(cls, value):
        from api import CITIES
        if value is not None and value not in CITIES:
            raise ValueError("Choose a city from /options")
        return value

    @field_validator("preferred_sectors")
    @classmethod
    def allowed_sectors(cls, values):
        from api import SECTORS
        if any(value not in SECTORS for value in values):
            raise ValueError("Choose investor sectors from /options")
        return values

    @field_validator("preferred_stages")
    @classmethod
    def allowed_stages(cls, values):
        from api import STAGES
        if any(value not in STAGES for value in values):
            raise ValueError("Choose investor stages from /options")
        return values

    @field_validator("preferred_cities")
    @classmethod
    def allowed_cities(cls, values):
        from api import CITIES
        if any(value not in CITIES for value in values):
            raise ValueError("Choose investor cities from /options")
        return values

    @model_validator(mode="after")
    def ticket_range_valid(self):
        if self.min_ticket_usd is not None and self.max_ticket_usd is not None and self.min_ticket_usd > self.max_ticket_usd:
            raise ValueError("Minimum ticket must not exceed maximum ticket")
        return self

class Login(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
class PitchDraft(BaseModel):
    recipient_id: int
    product: Optional[str] = Field(default=None, max_length=500)
    traction: Optional[str] = Field(default=None, max_length=300)
    funding_ask: Optional[str] = Field(default=None, max_length=120)
    milestone: Optional[str] = Field(default=None, max_length=200)
    why_fit: Optional[str] = Field(default=None, max_length=300)
class MemberMatchInput(BaseModel):
    sector: str
    stage: str
    city: str
class InvestorPreferences(BaseModel):
    preferred_sectors: list[str] = Field(default_factory=list, max_length=14)
    preferred_stages: list[str] = Field(default_factory=list, max_length=7)
    preferred_cities: list[str] = Field(default_factory=list, max_length=11)
    min_ticket_usd: Optional[float] = Field(default=None, ge=0)
    max_ticket_usd: Optional[float] = Field(default=None, ge=0)
    thesis: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_preferences(self):
        from api import SECTORS, STAGES, CITIES
        if any(value not in SECTORS for value in self.preferred_sectors):
            raise ValueError("Choose sectors from /options")
        if any(value not in STAGES for value in self.preferred_stages):
            raise ValueError("Choose stages from /options")
        if any(value not in CITIES for value in self.preferred_cities):
            raise ValueError("Choose cities from /options")
        if self.min_ticket_usd is not None and self.max_ticket_usd is not None and self.min_ticket_usd > self.max_ticket_usd:
            raise ValueError("Minimum ticket must not exceed maximum ticket")
        return self

class SaveStartup(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    company_domain: Optional[str] = Field(default=None, max_length=200)
    details: dict
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
        if body.role=='startup': c.execute('INSERT INTO startup_profiles(user_id,company,sector,stage,city,summary,contact_email,amount_usd,company_age) VALUES(?,?,?,?,?,?,?,?,?)',(uid,body.company,body.sector,body.stage,body.city,body.summary,body.email.lower(),body.amount_usd,body.company_age))
        else:
            sectors = body.preferred_sectors or ([body.sector] if body.sector else [])
            stages = body.preferred_stages or ([body.stage] if body.stage else [])
            cities = body.preferred_cities or ([body.city] if body.city else [])
            c.execute('INSERT INTO investor_profiles(user_id,name,sectors,stages,cities,summary,contact_email,min_ticket_usd,max_ticket_usd) VALUES(?,?,?,?,?,?,?,?,?)',
                      (uid,body.name,'|'.join(sectors),'|'.join(stages),'|'.join(cities),body.summary,body.email.lower(),body.min_ticket_usd,body.max_ticket_usd))
    return {'id':uid,'email':body.email.lower(),'role':body.role}

@router.post('/login')
def login(body: Login):
    email = body.email.lower()
    with db() as c:
        attempt = c.execute('SELECT * FROM login_attempts WHERE email=?', (email,)).fetchone()
        if attempt and attempt['blocked_until'] and attempt['blocked_until'] > stamp():
            fail(429, 'Too many login attempts. Try again later.')
        row = c.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    valid = False
    if row:
        stored, salt = row['password_hash'].split(':')
        valid = secrets.compare_digest(hashlib.scrypt(body.password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex(), stored)
    if not valid:
        with db() as c:
            prior = c.execute('SELECT failures FROM login_attempts WHERE email=?', (email,)).fetchone()
            failures = (prior['failures'] if prior else 0) + 1
            blocked_until = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat() if failures >= 5 else None
            c.execute('INSERT OR REPLACE INTO login_attempts VALUES(?,?,?)', (email, failures, blocked_until))
        fail(401, 'Invalid credentials')
    with db() as c:
        c.execute('DELETE FROM login_attempts WHERE email=?', (email,))
    token=secrets.token_urlsafe(32); expiry=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    with db() as c: c.execute('INSERT INTO tokens VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),row['id'],expiry))
    return {'access_token':token,'token_type':'bearer','expires_at':expiry,'role':row['role'],'email':row['email']}

@router.get('/me')
def me(user=Depends(current_user)):
    return {'id': user['id'], 'email': user['email'], 'role': user['role']}

def target_profile(c, user):
    if user['role']=='startup': return c.execute('SELECT * FROM startup_profiles WHERE user_id=?',(user['id'],)).fetchone()
    return c.execute('SELECT * FROM investor_profiles WHERE user_id=?',(user['id'],)).fetchone()

@router.get('/discover')
def discover(user=Depends(current_user)):
    """Find registered users of the opposite role, ranked by profile overlap."""
    with db() as c:
        own = target_profile(c, user)
        if not own:
            return []
        target_role = 'investor' if user['role'] == 'startup' else 'startup'
        table = 'investor_profiles' if target_role == 'investor' else 'startup_profiles'
        rows = c.execute(
            f'SELECT u.id, p.* FROM users u JOIN {table} p ON p.user_id=u.id WHERE u.role=? AND u.id<>?',
            (target_role, user['id']),
        ).fetchall()
    own_terms = {str(own[k]).strip().casefold() for k in ('sector', 'stage', 'city') if own[k]}
    results = []
    for row in rows:
        profile_terms = {str(row[k]).strip().casefold() for k in ('sector', 'stage', 'city') if row[k]}
        overlap = len(own_terms & profile_terms)
        score = round(100 * overlap / max(len(own_terms), 1), 1)
        results.append({'recipient_id': row['id'], 'role': target_role,
                        'name': row['name'] if target_role == 'investor' else row['company'],
                        'sector': row['sector'], 'stage': row['stage'], 'city': row['city'],
                        'summary': row['summary'], 'score': score})
    results.sort(key=lambda r: (-r['score'], (r['name'] or '').casefold()))
    return results

@router.post('/matches')
def matched_investors(body: MemberMatchInput, user=Depends(current_user)):
    """Rank actual investor accounts so every displayed pitch action has a recipient."""
    if user['role'] != 'startup':
        fail(403, 'Startup role required')
    chosen = {'sector': body.sector.casefold(), 'stage': body.stage.casefold(), 'city': body.city.casefold()}
    with db() as c:
        rows = c.execute(
            "SELECT u.id, p.name, p.sectors, p.stages, p.cities, p.summary FROM users u "
            "JOIN investor_profiles p ON p.user_id=u.id WHERE u.role='investor' AND u.id<>?",
            (user['id'],),
        ).fetchall()
    results = []
    for row in rows:
        profile = {'sector': row['sectors'], 'stage': row['stages'], 'city': row['cities']}
        matched = [key for key, value in chosen.items() if value and profile[key] and value in str(profile[key]).casefold()]
        results.append({
            'recipient_id': row['id'], 'name': row['name'], 'sector': row['sectors'],
            'stage': row['stages'], 'city': row['cities'], 'summary': row['summary'],
            'score': round(100 * len(matched) / max(sum(bool(value) for value in chosen.values()), 1), 1),
            'matched_on': matched,
        })
    results.sort(key=lambda item: (-item['score'], (item['name'] or '').casefold()))
    return results

@router.get('/investor/preferences')
def get_investor_preferences(user=Depends(current_user)):
    if user['role'] != 'investor':
        fail(403, 'Investor role required')
    with db() as c:
        row = c.execute('SELECT * FROM investor_profiles WHERE user_id=?', (user['id'],)).fetchone()
    return {
        'preferred_sectors': [value for value in (row['sectors'] or '').split('|') if value],
        'preferred_stages': [value for value in (row['stages'] or '').split('|') if value],
        'preferred_cities': [value for value in (row['cities'] or '').split('|') if value],
        'min_ticket_usd': row['min_ticket_usd'],
        'max_ticket_usd': row['max_ticket_usd'],
        'thesis': row['summary'] or '',
    }

@router.put('/investor/preferences')
def save_investor_preferences(body: InvestorPreferences, user=Depends(current_user)):
    if user['role'] != 'investor':
        fail(403, 'Investor role required')
    with db() as c:
        c.execute(
            'UPDATE investor_profiles SET sectors=?,stages=?,cities=?,summary=?,min_ticket_usd=?,max_ticket_usd=? WHERE user_id=?',
            ('|'.join(body.preferred_sectors), '|'.join(body.preferred_stages), '|'.join(body.preferred_cities),
             body.thesis, body.min_ticket_usd, body.max_ticket_usd, user['id']),
        )
    return {'saved': True}

@router.get('/investor/watchlist')
def get_investor_watchlist(user=Depends(current_user)):
    if user['role'] != 'investor':
        fail(403, 'Investor role required')
    with db() as c:
        rows = c.execute('SELECT * FROM investor_saved_startups WHERE user_id=? ORDER BY saved_at DESC', (user['id'],)).fetchall()
    return [{'id': row['id'], 'saved_at': row['saved_at'], **json.loads(row['details_json'])} for row in rows]

@router.post('/investor/watchlist', status_code=201)
def save_startup(body: SaveStartup, user=Depends(current_user)):
    if user['role'] != 'investor':
        fail(403, 'Investor role required')
    details = dict(body.details)
    details['company_name'] = body.company_name
    details['company_domain'] = body.company_domain
    try:
        with db() as c:
            cur = c.execute(
                'INSERT INTO investor_saved_startups(user_id,company_name,company_domain,details_json,saved_at) VALUES(?,?,?,?,?)',
                (user['id'], body.company_name, body.company_domain, json.dumps(details), stamp()),
            )
    except sqlite3.IntegrityError:
        fail(409, 'This startup is already on your watchlist')
    return {'id': cur.lastrowid, 'saved': True}

@router.delete('/investor/watchlist/{saved_id}')
def remove_saved_startup(saved_id: int, user=Depends(current_user)):
    if user['role'] != 'investor':
        fail(403, 'Investor role required')
    with db() as c:
        cur = c.execute('DELETE FROM investor_saved_startups WHERE id=? AND user_id=?', (saved_id, user['id']))
    if not cur.rowcount:
        fail(404, 'Saved startup not found')
    return {'id': saved_id, 'removed': True}

@router.post('/generate')
def generate(body: PitchDraft, user=Depends(current_user)):
    with db() as c:
        target=c.execute('SELECT * FROM users WHERE id=?',(body.recipient_id,)).fetchone()
        if not target or target['role']==user['role']: fail(404,'Opposite-role recipient not found')
        own=target_profile(c,user)
        other=(c.execute('SELECT * FROM investor_profiles WHERE user_id=?',(target['id'],)).fetchone() if target['role']=='investor' else c.execute('SELECT * FROM startup_profiles WHERE user_id=?',(target['id'],)).fetchone())
    if user['role']=='startup':
        company=own['company']; name=other['name']; sector=own['sector']
        parts = [f"Hi {name}, I'm {company}."]
        if body.product:
            parts.append(body.product.strip())
        if body.traction:
            parts.append(f"Traction: {body.traction.strip()}")
        if body.funding_ask:
            ask = f"We're raising {body.funding_ask.strip()}"
            if body.milestone:
                ask += f" to {body.milestone.strip()}"
            parts.append(ask + ".")
        if body.why_fit:
            parts.append(body.why_fit.strip())
        elif sector:
            parts.append(f"Your focus in {sector} feels aligned.")
        parts.append("Would you be open to a conversation?")
        msg = " ".join(parts)[:300].rstrip()
        if len(" ".join(parts)) > 300:
            msg = msg.rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
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
                amount_bin=('<$100K' if own['amount_usd'] is not None and own['amount_usd'] < 100_000 else '$100K-1M' if own['amount_usd'] is not None and own['amount_usd'] < 1_000_000 else '$1M-5M' if own['amount_usd'] is not None and own['amount_usd'] < 5_000_000 else '$5M-20M' if own['amount_usd'] is not None and own['amount_usd'] < 20_000_000 else '$20M-100M' if own['amount_usd'] is not None and own['amount_usd'] < 100_000_000 else '$100M+' if own['amount_usd'] is not None else 'Undisclosed'),
                amount_log10=__import__('math').log10(own['amount_usd'] + 1) if own['amount_usd'] is not None else None,
                company_age=own['company_age'] or 0)
            ranked=api.get_ranker(request).scores(__import__('pandas').Series(request.model_dump(exclude={'top_k'})))
            matched = next(((i, float(value)) for i, (n, value) in enumerate(ranked) if str(n).casefold()==investor_name.casefold()),None)
            if matched is not None:
                position, raw_score = matched
                score = (1 - position / max(len(ranked) - 1, 1)) if len(ranked) > 1 else max(0.0, min(1.0, raw_score))
            if score is not None:
                profile=api.read_csv_or_empty(api.INVESTOR_PROFILES_PATH)
                detail=api.profile_details(investor_name,profile)
                sectors=str(detail.get('top_sectors') or '')
                if own['sector'] and own['sector'] in sectors: reasons.append(f"Investor's portfolio includes {own['sector']}")
                if own['stage'] and own['stage'] in str(detail.get('top_stages') or ''): reasons.append(f"Investor has backed {own['stage']} companies")
                if not reasons: reasons.append('Score from trained investor ranking model')
        except Exception:
            score=None
    if user['role'] == 'startup' and score is None:
        # Registered profiles remain matchable even when their fund name is absent
        # from the historical investor dataset.
        pairs = [(own['sector'], other['sectors']), (own['stage'], other['stages']), (own['city'], other['cities'])]
        matching = [label for label, left, right in zip(('sector', 'stage', 'city'), (p[0] for p in pairs), (p[1] for p in pairs))
                    if left and right and str(left).casefold() in str(right).casefold()]
        score = len(matching) / max(sum(bool(left) for left, _ in pairs), 1)
        reasons.extend([f"Shared {item} focus" for item in matching])
        if not matching:
            reasons.append('Profile-based introduction; no historical name match')
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
