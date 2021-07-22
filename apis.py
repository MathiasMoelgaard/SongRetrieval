import urllib.parse
import base64
import requests
from bs4 import BeautifulSoup

# Common methods used by both APIs
class API:
    
    def get_auth_header(self):
        raise NotImplementedError()  # sublcasses must implement authentication
    
    def post(self, endpoint, **req_params):
        r = requests.post(self.base_url + endpoint, data=req_params,
                          headers={"Authorization": self.get_auth_header()})
        r.raise_for_status()
        return r.json()
    
    def get(self, endpoint, **req_params):
        r = requests.get(self.base_url + endpoint, params=req_params,
                         headers={"Authorization": self.get_auth_header()})
        r.raise_for_status()
        return r.json()


# Simple Spotify API Client
class Spotify(API):
    
    client_id = "d85782c7aa594f5eb790a4f0bf4af022"
    client_secret = "19ae97240a664cd0a17909011d874091"
    base_url = "https://api.spotify.com/v1/"
    
    def __init__(self):
        self.access_token = None
        self.access_token_type = None
    
    def get_auth_header(self):
        if not self.access_token:
            raise Exception("Not authenticated; call authenticate() first")
        return "{} {}".format(self.access_token_type.title(), self.access_token)
    
    def authenticate(self):
        basic_header = base64.b64encode("{}:{}"
                                        .format(self.client_id, self.client_secret)
                                        .encode("ascii")).decode("ascii")
        headers = {"Authorization": "Basic {}".format(basic_header)}
        data = {"grant_type": "client_credentials"}
        r = requests.post("https://accounts.spotify.com/api/token", data=data, headers=headers)
        r.raise_for_status()
        res = r.json()
        if "access_token" not in res or "token_type" not in res:
            raise Exception("Malformed Spotify API response")
        self.access_token = res["access_token"]
        self.access_token_type = res["token_type"]


# Simple Genius API Client
class Genius(API):
    
    client_id = "HDGdy8grlfzttb5mZx3hKJzLdgJdlTnyxGTPY06Ldkrx21MCkug1Vi6UQwgW4KXW"
    client_secret = "wiNgOjrAorE4Aomx68f1myCOnKwx4GEQnsVksJrOCcb_VYNnsbeVa0gRcCUgRv9Ox8n2Vigip6DwQiwkLOYAQg"
    redirect_uri = "http://localhost"
    base_url = "https://api.genius.com/"
    
    def __init__(self):
        self.access_token = None
    
    def get_auth_header(self):
        return "Bearer {}".format(self.access_token)
    
    def authenticate(self, scope=""):
        # Ask user for authorization
        base_url = "https://api.genius.com/oauth/authorize"
        params = {"client_id": self.client_id,
                  "redirect_uri": self.redirect_uri,
                  "scope": scope,
                  "state": "",
                  "response_type": "code"}
        query_str = urllib.parse.urlencode(params)
        link_url = "https://api.genius.com/oauth/authorize?" + query_str
        print("Please follow this link and authorize the app:")
        print(link_url + "\n")
        print("Then, paste the URL you are redirected to here: ")
        redirect_url = input()
        redirect_url_parsed = urllib.parse.urlparse(redirect_url)
        redirect_params = urllib.parse.parse_qs(redirect_url_parsed.query)
        if "code" not in redirect_params:
            raise Exception("Malformed OAuth response")
        
        # Get access token from user code
        token_req_params = {"code": redirect_params["code"],
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "grant_type": "authorization_code",
                            "redirect_uri": self.redirect_uri,
                            "response_type": "code"}
        r = requests.post("https://api.genius.com/oauth/token",
                          data=token_req_params)
        r.raise_for_status()
        res = r.json()
        if "access_token" not in res:
            raise Exception("Malformed OAuth token request response")
        self.access_token = res["access_token"]
        print("Authentication successful.")

    def lyrics_from_song_api_path(self, song_api_path):
        # adapted from https://bigishdata.com/2016/09/27/getting-song-lyrics-from-geniuss-api-scraping/
        page_url = "http://genius.com" + song_api_path
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, "html.parser")
        [h.extract() for h in html('script')]
        lyrics = html.find("div", class_="lyrics").get_text()
        return lyrics
