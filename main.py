# import requirements needed
from flask import Flask, render_template, request
from utils import get_base_url
from train import nba_knn

prediction = None
model = None
winner = None

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12380
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}', methods = ["GET", "POST"])
def home():

    global prediction
    global model
    global winner
  
    if request.method == 'POST':
      VisitorTeamName = request.form.get("vname") # find user input
      HomeTeamName = request.form.get("hname") # find user input
      prediction = model.classify(VisitorTeamName, HomeTeamName)

      # Output the winning team instead of binary value
      if prediction == 1:
        winner = HomeTeamName
      elif prediction == 0:
        winner = VisitorTeamName
        
    return render_template("index.html", prediction = winner)  
  
# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    model = nba_knn()
    app.run(host = '0.0.0.0', port=port, debug=True)