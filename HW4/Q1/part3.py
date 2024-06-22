import hashlib


def hashed_password(password):
    return hashlib.sha512(password.encode())

def get_combination(list_of_str):
    result = []
    for str in list_of_str:
        list_copied = list_of_str.copy()
        list_copied.remove(str)
        result.append(str + list_copied[0] + list_copied[1])
        result.append(str + list_copied[1] + list_copied[0])
    return result

if __name__ == "__main__":

    digital_corp = open("keystreching-digitalcorp.txt", 'r')
    rock_you = open("rockyou.txt", 'r')

    for digi_line in digital_corp.readlines()[1:]:
        digi_line = digi_line.split(",")
        username = digi_line[0]
        password_salt = digi_line[1]
        user_hashed_password = digi_line[2].replace("\n", "")
        for i in range(6):
            for dict_line in rock_you.readlines():
                plain_password = dict_line.replace("\n", "")
                prev_hash = ""
                for _ in range(2000):
                        str_elements = [prev_hash, plain_password, password_salt]
                        combinations = get_combination(str_elements)
                        hash_x = hashed_password(combinations[i]).hexdigest()

                        if hash_x == user_hashed_password:
                            print(username + " password is " + plain_password)
                            break
                        prev_hash = hash_x
            rock_you.seek(0)

    digital_corp.close()
    rock_you.close()