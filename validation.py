
crud_username = "affektive"
crud_password = 'affektive13371'

missing_params = "missing parameter(s): "

def login_check(username, password):
	return username==crud_username and password==crud_password;

def errors_login_params(keys):
	if not 'username' in keys and 'password' in keys:
		return jsonify(error=missing_params +'username or password')
	return ''
		
def errors_insert_params(keys):
	errors = errors_login_params(keys)
	if errors:
		return errors
	if not ('data' in keys):
		return jsonify(error=missing_params + 'data')
	return ''
		

def errors_insert_data_params(keys):
	if not ('timestamp' in keys and 'hr' in keys and 'gsr' in keys and 'state' in keys):
		return jsonify(error='missing parameter(s)')
	return ''

