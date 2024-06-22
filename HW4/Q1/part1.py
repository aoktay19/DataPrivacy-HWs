import hashlib


def hashed_password(password):
    return hashlib.sha512(password.encode())


def generate_dict_of_passwords():

    rockyou_file = open("rockyou.txt", 'r')

    with open('part1_passwords.txt', 'w') as file:

        for password in rockyou_file.readlines():
            password = password.replace("\n", "")
            hashed_password_rock = hashed_password(password).hexdigest()
            line = f"{password},{hashed_password_rock}\n"
            file.write(line)

        rockyou_file.close()
        file.close()

if __name__ == "__main__":
    generate_dict_of_passwords()

    digitalcorp = open("digitalcorp.txt", 'r')
    password_dictionary = open("part1_passwords.txt", 'r')

    for digi_line in digitalcorp.readlines()[1:]:
        digi_line = digi_line.split(",")
        username = digi_line[0]
        user_hashed_password = digi_line[1].replace("\n", "")

        for dict_line in password_dictionary.readlines():
            dict_line = dict_line.split(",")
            plain_password = dict_line[0]
            password_after_hash = dict_line[1].replace("\n", "")

            if password_after_hash == user_hashed_password:
                print(username + " password is " + plain_password)
                break

        password_dictionary.seek(0)