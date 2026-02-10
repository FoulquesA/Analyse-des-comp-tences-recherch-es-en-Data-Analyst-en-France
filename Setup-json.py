import requests
import json
import time

CLIENT_ID = "PUT_YOUR_CLIENT_ID_HERE"  # Remplacez par votre client_id
CLIENT_SECRET = "PUT_YOUR_CLIENT_SECRET_HERE" # Remplacez par votre client_secret

def get_token(client_id, client_secret):
    url = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "api_offresdemploiv2 o2dsoffre"
    }
    response = requests.post(url, data=data)
    
    if response.status_code != 200:
        print(f"Erreur token : {response.status_code} - {response.text}")
        return None
    
    return response.json().get("access_token")

def get_offres_analyste(token):
    toutes_offres = []
    debut = 0
    taille_page = 150
    
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    
    while True:
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "motsCles": "analyste",  # filtre pour les offres
            "range": f"{debut}-{debut + taille_page - 1}"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code not in [200, 206]:  # 206 = Partial Content (normal pour pagination)
            print(f"Erreur API : {response.status_code} - {response.text}")
            break
        
        data = response.json()
        offres = data.get("resultats", [])
        
        if not offres:
            break
        
        toutes_offres.extend(offres)
        print(f"Récupéré {len(toutes_offres)} offres au total...")
        
        if len(offres) < taille_page:
            break
            
        debut += taille_page
        time.sleep(0.15)
    
    return toutes_offres


def save_offres_json(offres, filename="offres_analyste.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(offres, f, ensure_ascii=False, indent=2)
    print(f"\n{len(offres)} offres sauvegardées dans {filename}")

if __name__ == "__main__":
    print("Connexion à l'API France Travail...\n")
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    
    if not token:
        exit("Impossible de récupérer le token.")
    
    print("Token obtenu. Récupération des offres avec 'analyste'...\n")
    offres = get_offres_analyste(token)
    
    if offres:
        save_offres_json(offres)
        
        print("\nPremiers intitulés récupérés :")
        for i, offre in enumerate(offres[:10], 1):
            print(f"{i}. {offre.get('intitule')}")
    else:
        print("Aucune offre trouvée")

