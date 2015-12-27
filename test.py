import requests
import json

r = requests.get('http://affektive.agif.me/api/measurement')
print r.json()
