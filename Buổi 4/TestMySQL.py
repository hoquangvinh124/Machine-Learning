import mysql.connector

server="localhost"
port=3306
database="studentmanagement"
username="root"
password="1234"

def get_connection():
    return mysql.connector.connect(
        host=server,
        port=port,
        database=database,
        user=username,
        password=password)

def print_students(dataset):
    print("Running function: print_students")
    align='{0:<3} {1:<6} {2:<15} {3:<10}'
    print(align.format('ID', 'Code','Name',"Age"))
    for item in dataset:
        id=item[0]
        code=item[1]
        name=item[2]
        age=item[3]
        print(align.format(id,code,name,age))

def select_all_students():
    print("Running function: select_all_students")
    conn = get_connection()
    cursor = conn.cursor()
    sql="select * from student"
    cursor.execute(sql)
    dataset=cursor.fetchall()
    print_students(dataset)
    cursor.close()
    conn.close()

def select_students_by_age(min_age, max_age, order_desc=False):
    print("Running function: select_students_by_age")
    conn = get_connection()
    cursor = conn.cursor()
    sql="SELECT * FROM student where Age>=%s and Age<=%s"
    if order_desc:
        sql += " order by Age desc"
    cursor.execute(sql, (min_age, max_age))
    dataset=cursor.fetchall()
    print_students(dataset)
    cursor.close()
    conn.close()

def select_student_by_id(student_id):
    print("Running function: select_student_by_id")
    conn = get_connection()
    cursor = conn.cursor()
    sql="SELECT * FROM student where ID=%s"
    cursor.execute(sql, (student_id,))
    dataset=cursor.fetchone()
    if dataset!=None:
        id,code,name,age,avatar,intro=dataset
        print("Id=",id)
        print("code=",code)
        print("name=",name)
        print("age=",age)
    cursor.close()
    conn.close()

def select_students_limit_offset(limit, offset):
    print("Running function: select_students_limit_offset")
    conn = get_connection()
    cursor = conn.cursor()
    sql="SELECT * FROM student LIMIT %s OFFSET %s"
    cursor.execute(sql, (limit, offset))
    dataset=cursor.fetchall()
    print_students(dataset)
    cursor.close()
    conn.close()

def paging_students(limit, step):
    print("Running function: paging_students")
    conn = get_connection()
    cursor = conn.cursor()
    sql="SELECT count(*) FROM student"
    cursor.execute(sql)
    dataset=cursor.fetchone()
    rowcount=dataset[0]
    for offset in range(0,rowcount,step):
        sql=f"SELECT * FROM student LIMIT {limit} OFFSET {offset}"
        cursor.execute(sql)
        dataset=cursor.fetchall()
        print_students(dataset)
    cursor.close()
    conn.close()

def insert_student(code, name, age):
    print("Running function: insert_student")
    conn = get_connection()
    cursor = conn.cursor()
    sql="insert into student (code,name,age) values (%s,%s,%s)"
    val=(code, name, age)
    cursor.execute(sql,val)
    conn.commit()
    print(cursor.rowcount," record inserted")
    cursor.close()
    conn.close()

def insert_students_bulk(values):
    print("Running function: insert_students_bulk")
    conn = get_connection()
    cursor = conn.cursor()
    sql="insert into student (code,name,age) values (%s,%s,%s)"
    cursor.executemany(sql,values)
    conn.commit()
    print(cursor.rowcount," record inserted")
    cursor.close()
    conn.close()

def update_student_name_by_code(new_name, code):
    print("Running function: update_student_name_by_code")
    conn = get_connection()
    cursor = conn.cursor()
    sql="update student set name=%s where Code=%s"
    val=(new_name, code)
    cursor.execute(sql,val)
    conn.commit()
    print(cursor.rowcount," record(s) affected")
    cursor.close()
    conn.close()

def delete_student_by_id(student_id):
    print("Running function: delete_student_by_id")
    conn = get_connection()
    cursor = conn.cursor()
    sql = "DELETE from student where ID=%s"
    val = (student_id,)
    cursor.execute(sql, val)
    conn.commit()
    print(cursor.rowcount," record(s) affected")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    select_all_students()
    select_students_by_age(22, 26)
    select_students_by_age(22, 26, order_desc=True)
    select_student_by_id(1)
    select_students_limit_offset(3, 0)
    select_students_limit_offset(3, 3)
    paging_students(limit=3, step=3)
    insert_student("sv07","Trần Duy Thanh",45)
    insert_students_bulk([
        ("sv08","Trần Quyết Chiến",19),
        ("sv09","Hồ Thắng",22),
        ("sv10","Hoàng Hà",25),
    ])
    update_student_name_by_code('Hoàng Lão Tà','sv09')
    delete_student_by_id(14)
    delete_student_by_id(13)