import pytest
from fastapi.testclient import TestClient
import pitch_requests as pitch
import api

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(pitch, 'DB_PATH', tmp_path / 'pitch-test.sqlite3')
    pitch.init_db()
    # The API lifespan normally loads the neural checkpoint. Pitch lifecycle
    # tests exercise auth/storage and should not deserialize ML weights.
    def unavailable_ranker():
        raise RuntimeError('model loading intentionally disabled in tests')
    monkeypatch.setattr(api, 'load_ranker', unavailable_ranker)
    with TestClient(api.app) as c:
        yield c

def test_register_login_validation_and_duplicate_email(client):
    assert client.post('/pitch/register',json={'email':'bad@example.com','password':'longpassword','role':'startup'}).status_code==422
    s=account(client,'startup','one@example.com',company='One')
    assert client.post('/pitch/register',json={'email':'one@example.com','password':'StrongPassword123','role':'startup','company':'One'}).status_code==409
    assert client.post('/pitch/login',json={'email':'one@example.com','password':'not-the-password'}).status_code==401
    assert client.get('/pitch/requests',headers={'Authorization':'Bearer invalid'}).status_code==401

def test_score_unavailable_is_null_and_never_zero_fabrication(client,monkeypatch):
    s,_=pair(client)
    monkeypatch.setattr(api,'ranker',None)
    monkeypatch.setattr(api,'load_ranker',lambda: (_ for _ in ()).throw(RuntimeError('missing')))
    draft=client.post('/pitch/generate',headers=s,json={'recipient_id':2})
    assert draft.status_code==200 and draft.json()['match_score'] is None
    sent=submit(client,s,2,'no fabricated score')
    assert sent.status_code==201 and sent.json()['match_score'] is None

def test_score_preserves_model_score(client,monkeypatch):
    s,_=pair(client)
    class Model:
        def scores(self,row): return [('North Fund',.63)]
    monkeypatch.setattr(api,'ranker',Model())
    draft=client.post('/pitch/generate',headers=s,json={'recipient_id':2}).json()
    assert draft['match_score']==pytest.approx(.63)

def account(c, role, email, **profile):
    data={'email':email,'password':'StrongPassword123','role':role,**profile}
    assert c.post('/pitch/register',json=data).status_code==201
    token=c.post('/pitch/login',json={'email':email,'password':data['password']}).json()['access_token']
    return {'Authorization':f'Bearer {token}'}

def pair(c):
    startup=account(c,'startup','founder@example.com',company='Acme',sector='Fintech',stage='Seed',city='Bengaluru')
    investor=account(c,'investor','fund@example.com',name='North Fund',sector='Fintech')
    return startup,investor

def submit(c,headers,recipient_id,message='Hello, let us connect',**kw):
    return c.post('/pitch/requests',headers=headers,json={'recipient_id':recipient_id,'message':message,**kw})

def test_startup_lifecycle_privacy_notifications_and_status(client):
    s,i=pair(client)
    draft=client.post('/pitch/generate',headers=s,json={'recipient_id':2}).json()
    assert draft['message'] and len(draft['message'])<=300
    assert draft['match_score'] is None or 0<=draft['match_score']<=1
    assert submit(client,s,2,draft['message'],match_score=.71,match_reasons=['Fintech fit']).status_code==201
    inbox=client.get('/pitch/inbox',headers=i).json(); assert inbox[0]['status']=='SENT' and inbox[0]['contact_email'] is None
    rid=inbox[0]['id']
    assert client.get(f'/pitch/requests/{rid}',headers=s).json()['contact_email'] is None
    assert client.post(f'/pitch/requests/{rid}/respond',headers=i,json={'action':'accept'}).json()['status']=='ACCEPTED'
    assert client.get(f'/pitch/requests/{rid}',headers=s).json()['contact_email']=='fund@example.com'
    assert client.get(f'/pitch/requests/{rid}',headers=i).json()['contact_email']=='founder@example.com'
    assert any('accepted' in n['message'] for n in client.get('/pitch/notifications',headers=s).json())

def test_investor_invitation_decline_info_reply_and_withdraw(client):
    s,i=pair(client)
    # Investor invite -> startup asks for info; status updates and sender receives notification.
    sent=submit(client,i,1,'Would like to meet').json(); rid=sent['id']
    assert client.post(f'/pitch/requests/{rid}/respond',headers=s,json={'action':'request_info','message':'Please share terms'}).json()['status']=='INFO_REQUESTED'
    assert client.get('/pitch/notifications',headers=i).json()
    assert client.post(f'/pitch/requests/{rid}/withdraw',headers=i).json()['status']=='WITHDRAWN'
    # New request can follow withdrawal; decline is terminal.
    rid=submit(client,i,1,'Second invite').json()['id']
    assert client.post(f'/pitch/requests/{rid}/respond',headers=s,json={'action':'decline'}).json()['status']=='DECLINED'
    assert client.post(f'/pitch/requests/{rid}/respond',headers=s,json={'action':'accept'}).status_code==409

def test_pending_duplicate_and_rolling_limit(client):
    s,_=pair(client)
    assert submit(client,s,2,'First').status_code==201
    assert submit(client,s,2,'Duplicate').status_code==409
    # Distinct recipients exercise the per-sender rolling-window count.
    for x in range(3,7): account(client,'investor',f'i{x}@example.com',name=f'Investor {x}')
    for x in range(3,7): assert submit(client,s,x,f'Message {x}').status_code==201
    account(client,'investor','i7@example.com',name='Investor 7')
    assert submit(client,s,7,'Sixth').status_code==429

def test_auth_role_validation_and_ownership(client):
    s,i=pair(client)
    rid=submit(client,s,2,'Hello').json()['id']
    other=account(client,'startup','other@example.com',company='OtherCo')
    assert client.get(f'/pitch/requests/{rid}',headers=other).status_code==404
    assert client.post(f'/pitch/requests/{rid}/respond',headers=other,json={'action':'accept'}).status_code==404
    assert client.get('/pitch/inbox',headers=s).status_code==403
    assert client.get('/pitch/requests').status_code==401
    assert submit(client,s,2,'x'*301).status_code==422

def test_decline_and_withdraw_terminal_transitions(client):
    s,i=pair(client)
    rid=submit(client,s,2,'withdraw me').json()['id']
    assert client.post(f'/pitch/requests/{rid}/withdraw',headers=s).json()['status']=='WITHDRAWN'
    assert client.post(f'/pitch/requests/{rid}/respond',headers=i,json={'action':'accept'}).status_code==409

def test_rank_and_startup_matching_routes_remain_available(client,monkeypatch):
    class Dummy:
        def scores(self,row): return [('North Fund',.8)]
    monkeypatch.setattr(api,'ranker',Dummy())
    assert client.post('/rank',json={'sector_group':'Fintech','stage_bucket':'Seed','city_group':'Bengaluru','amount_bin':'$1M-5M','company_age':2}).status_code==200
    result=client.post('/match/startups',json={'preferred_sectors':['Fintech'],'top_k':3})
    assert result.status_code==200
    assert 'matched_startups' in result.json()
