import hashlib


def hashed_password(password):
    return hashlib.sha512(password.encode())

if __name__ == "__main__":

    digital_corp = open("salty-digitalcorp.txt", 'r')
    rock_you = open("rockyou.txt", 'r')

    for digi_line in digital_corp.readlines()[1:]:
        digi_line = digi_line.split(",")
        username = digi_line[0]
        password_salt = digi_line[1]
        user_hashed_password = digi_line[2].replace("\n", "")

        for dict_line in rock_you.readlines():
            plain_password = dict_line.replace("\n", "")
            if hashed_password(plain_password + password_salt).hexdigest() == user_hashed_password:
                print(username + " password is " + plain_password)
                break
            if hashed_password(password_salt + plain_password).hexdigest() == user_hashed_password:
                print(username + " password is " + plain_password)
                break
        rock_you.seek(0)