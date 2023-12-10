import pymysql
db = pymysql.connect(host='localhost',
                     user='root',
                     password='155955',
                     database='picture')
 
# 创建一个游标对象
cursor = db.cursor()
 
cursor.execute("SELECT * FROM Staffs")
while True:
    singleData = cursor.fetchone()
    if singleData is None:
        break
    print(singleData[1])
    print(singleData[2])
   
    
