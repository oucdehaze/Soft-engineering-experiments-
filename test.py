from numpy import insert
import pymysql
username = "def"
password = "456"
db = pymysql.connect(host='localhost',
                     user='root',
                     password='155955',
                     database='picture')
cursor = db.cursor()
cursor.execute("insert into Staffs values( '%s',  '%s')" % (username, password))
db.commit()