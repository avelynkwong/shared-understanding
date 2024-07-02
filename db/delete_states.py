import mysql.connector
from get_secrets import get_secret

mysql_secrets = get_secret("mysql_secrets")

config = {
    "user": mysql_secrets["username"],
    "password": mysql_secrets["password"],
    "host": mysql_secrets["host"],
    "port": mysql_secrets["port"],
}
DB_NAME = mysql_secrets["dbInstanceIdentifier"]

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()

cursor.execute("USE {}".format(DB_NAME))
cursor.execute(f"DELETE FROM {DB_NAME}.states")
cnx.commit()
print("States deleted")

cursor.close()
cnx.close()
