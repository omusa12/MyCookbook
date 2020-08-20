import psycopg2 as ps
# define credentials 
credentials = {'POSTGRES_ADDRESS' : 'test-db.cwtu2obx88su.us-east-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : '5432', # change to your port
               'POSTGRES_USERNAME' : 'postgres', # change to your username
               'POSTGRES_PASSWORD' : 'Sally123$', # change to your password
               'POSTGRES_DBNAME' : 'test-db'} # change to your db name
# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])
cur = conn.cursor()

query = """SELECT * FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog'
            AND schemaname != 'information_schema';"""
cur.execute(query)
cur.fetchall()

cur.execute("""CREATE TABLE table_1
                (column_1 integer, 
                column_2 float,
                column_3 varchar(50),
                column_4 boolean);""")
# Commit table creation
conn.commit()