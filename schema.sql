create table if not exists entries (
	id integer primary key autoincrement,
	userid integer not null,
	timestamp date not null,
	hr integer not null,
	gsr integer not null,
	state text not null,
	level integer not null
);
