import requests
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY="dental_scan_ai_secret_key_2025_change_this"
ALGORITHM="HS256"
token = jwt.encode({"sub": "gopi123@gmail.com", "exp": datetime.utcnow() + timedelta(minutes=1000)}, SECRET_KEY, algorithm=ALGORITHM)

headers = {"Authorization": f"Bearer {token}"}
base_url = "http://localhost:8000"

print("Fetching patients...")
r = requests.get(f"{base_url}/patients", headers=headers)
print("Patients Status:", r.status_code)
if r.status_code == 200:
    patients = r.json()
    print("Found patients:", len(patients))
    for p in patients:
        print(f"Patient ID: {p['id']}, Name: {p['name']}")
        print(f"Fetching scans for {p['id']}...")
        r_scan = requests.get(f"{base_url}/scans/{p['id']}", headers=headers)
        print("  Scan Status:", r_scan.status_code)
        if r_scan.status_code == 200:
            scans = r_scan.json()
            print(f"  Found scans: {len(scans)}")
        else:
            print("  Scan Error:", r_scan.text)
else:
    print("Patients Error:", r.text)
