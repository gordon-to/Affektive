
crud_username = "affektive"
crud_password = 'affektive13371'

def login_check(username, password):
	return username==crud_username and password==crud_password;

def check_get_request(keys):
	errors = ''
	if not 'username' in keys and 'password' in keys:
		errors = jsonify(error='missing parameter(s)')
	return errors
		
def check_insert_request(keys):
	errors = ''
	if not ('username' in keys and 'password' in keys and 'timestamp' in keys and 'gsr' in keys and 'state' in keys):
		errors = jsonify(error='missing parameter(s)')
	return errors

