def shift(current_list, parameters_of_polynomial):
    """ A new value is inserted on the left side of the list (value_to_add)
        and the most right value is the bit_out that is returning to the caller
        function alongside the new list. value_to_add is the xor operation between
        x^10 and x^3.
        This function is executed on every run and returns to the caller the new list, 
        which is the initial value for the next run and the outstrem bit
    """
    value_to_add = current_list[parameters_of_polynomial[1]-1] ^ current_list[parameters_of_polynomial[0]-1]
    current_list.insert(0, value_to_add)
    bit_out = current_list.pop()
    return current_list, bit_out

def calc_output_stream(current_list, runs=15, parameters_of_polynomial = 0):
    """ The default number of runs is 1023 because of the grade of polynomal. 
    L=10, so the period of the output stream is 2^10 - 1 = 1023.
    On every run a shift() operation is performed on the list
    and the result is inserted in the list. In order to handle the output bit 
    in an efficient way, every bit_out value is inserted in a new list 
    called output stream. The list is printed on screen and in a file named "out.txt" 
    """
    output_stream=[]
    print("Итерация  |  Состояние   |  Выходной бит")
    for i in range(runs):                       
        current_list, feedback = shift(current_list, parameters_of_polynomial)
        if i > 9:
          print(f"{i}        |   {current_list}  |    {feedback}")
        else:
          print(f"{i}         |   {current_list}  |    {feedback}")
        output_stream.append(feedback)  
    #print(output_stream)
    print()
    print("Output: ",end=" ")
    print("".join(str(x) for x in output_stream))
    create_output_stream_file("out.txt", output_stream)

def create_output_stream_file(filename, out_stream):
    with open(filename, 'w') as out:        
        out.write("".join(str(x) for x in out_stream))

def mod(num, a):
 
    # Initialize result
    res = 0
 
    # One by one process all digits
    # of 'num'
    for i in range(0, len(num)):
        res = (res * 10 + int(num[i])) % a
 
    return res

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x

def backpack_find_A(B, arr, arr2, summa, symbolK):
  for i in range(max(B), 2*(max(B))+ 1):
    for j in range(2,i):
      if gcd(j,i) == 1 :
        for m in range(len(B)):
          arr[m] = mod(str(j*B[m]), i)
        if all(arr[i-1]<arr[i] for i in range(1, len(arr))) == True :
          # print(f"Value u = {j} and  value m = {i}")
          t = pow(j, -1, i)
          if t <= max(B):
            print(f"1) Выбрали m = {i}")
            print(f"2) Выбрали u = {j}, где {j} < {i} и НОД ({j},{i}) = 1")
            print(f"3) Вектор A = {arr}")
            print(f"4) Нашли мультипликативно обратное к u, t = {t}") 
            for l in range(len(arr) - 1):
              summa +=(t*arr[l]) // i
            summa2 = ((t*arr[0])//i) + ((t*arr[1]) // i)

            if summa2 < ((t*arr[2]) // i):
              print("1st cond")

            if i <= sum(arr):
              print('2nd cond')

            if (summa2 + (t*arr[2]) // i) < t:
              print("3rd cond")
            # if (summa + t*arr[2]//i) < t:
            #   print(f"Нашли какой-то стремный вектор, который лучше взять {arr}")
            if summa < arr[2]:
              print(f"A(k) = ({arr[0] + [t*arr[0]//i if t*arr[0]//i > 0 else 0][0]}, {arr[1] + [t*arr[1]//i if t*arr[1]//i > 0 else 0][0]}, {arr[2] + [t*arr[2]//i if t*arr[2]//i > 0 else 0][0]})")
              print(f"(A(k),t,m+kt) = (({str(arr[0]) + str([' + ' + str(t*arr[0]//i) + symbolK if t*arr[0]//i > 0 else ''][0])}, {str(arr[1]) + str([' + ' + str(t*arr[1]//i) + symbolK if t*arr[1]//i > 0 else ''][0])}, {str(arr[2]) + str([' + ' + str(t*arr[2]//i) + symbolK if t*arr[2]//i > 0 else ''][0])}),{t}, {i + t})")
              print()
              print("Проверка")
              print()
              print(f"module = {i+t}")
              for s in range(len(B)):
                arr2[s] = mod(str(t * (arr[s] + [t*arr[s]//i if t*arr[s]//i > 0 else 0][0])), i + t)
              print(arr2)
              print("--------------------------------")
              summa = 0
            else:
              summa = 0


# ----------

def generate_first_number(value):
    znaleziona = False
    i = value + 1
    while not (znaleziona):
        pierwsza = True
        for n in range(2, i):
            if i % n == 0:
                pierwsza = False
                break
        if i % 4 == 3:
            znaleziona = True
        else:
            i += 1
    return i


def generate_next_number(value, m):
    return value ** 2 % m

def generate_next_numb(value, m, e):
    return value ** e % m 

def BBS():
    p = generate_first_number(1450)
    q = generate_first_number(p)
    n = p * q
    number = 32562359
    bitArray = []
    outputArray = []
    howManyZero = 0
    howManyOne = 0
    print (f"Пусть p = {p}, q = {q}")
    print(f"Число Блюма N = {n}")
    print(f"Целое число x = {number}")
    for i in range(11): # количество бит(длина в текущем примере равна 10-битной)
        number = generate_next_number(number, n)
        bitArray = list('{0:0b}'.format(number))
        print(f"Число {i}: {number},  младший бит {bitArray[len(bitArray) - 1]}")
        if bitArray[len(bitArray) - 1] == '0':
            howManyZero += 1
        if bitArray[len(bitArray) - 1] == '1':
            howManyOne += 1
        outputArray.append(bitArray[len(bitArray) - 1])
    print (outputArray)
    print ("Zero:", howManyZero)
    print ("One:", howManyOne)


def gcd(a, b):
    # Return the GCD of a and b using Euclid's Algorithm
    while a != 0:
        a, b = b % a, a
    return b

from random import randint, randrange

def ismillerprime( n, k):
        if n == 1:
            return False
        if n in [2, 3, 5, 7, 11, 13, 17, 19]:
            return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            if n % p == 0:
                return False
        r = 0
        s = n - 1
        while s % 2 == 0:
            r += 1
            s //= 2
        for i in range(k):
            a = randrange(2, n - 1)
            x = pow(a, s, n)
            if x == 1 or x == n - 1:
                continue
            for j in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

def randomprime(length, millerrounds):
        length -= 1
        start = randint(10 ** length, 10 ** length * 9)
        if start % 2 == 0:
            start += 1
        counter = 0
        isloop = True
        while isloop:
            testnumber = start + (counter * 2)
            if ismillerprime(testnumber, millerrounds):
                isloop = False
            counter += 1
        return testnumber

def RSA():
  p = randomprime(length = 6 // 2, millerrounds = 2)
  print(f"Значение p = {p}")
  q = randomprime(length = 6 // 2, millerrounds = 2)
  print(f"Значение q = {q}")
  n = p * q
  print(f"n = {n}")
  phi = (p-1)*(q-1)
  print(f"phi(n) = {phi}")
  e = 0
  bitArray = []
  outputArray = []
  for i in range(40):
    if gcd(phi, i) == 1:
      e = i
  print(f"e = {e}")
  start_x = 14 # стартовое число генератора x0
  print(f"Случайное стартовое целое число x0 = {start_x}")
  if start_x % n:
    outputArray.append(0)
  else:
    outputArray.append(1)
  howManyZero = 0
  howManyOne = 0
  print(f"Число 0: {start_x},  младший бит {outputArray[0]}")
  for i in range(10): # количество бит(длина в текущем примере равна 10-битной)
        start_x = generate_next_numb(start_x, n, e)
        bitArray = list('{0:0b}'.format(start_x))
        print(f"Число {i+1}: {start_x},  младший бит {bitArray[len(bitArray) - 1]}")
        if bitArray[len(bitArray) - 1] == '0':
            howManyZero += 1
        if bitArray[len(bitArray) - 1] == '1':
            howManyOne += 1
        outputArray.append(bitArray[len(bitArray) - 1])
  print (outputArray)
  # print(f"p = {p}")
  # print(f"q = {q}")
  # print(f"n = {n}")
  print ("Zero:", howManyZero)  
  print ("One:", howManyOne)


def simple_bm(flow, **arg):
    n, l, fx = 0, [], []
  
    '''
    flow = [0, 1, 0, 1, 0, 1, 0]
    n=2 -> a(n-1)=a(1)=1
    '''
    for i in flow:
        if i == 1: break
        n += 1
    '''
    >> i=0 i=1 >>
    l(0)=l(1)=0
    fx(0)=fx(1)=1
    '''
    for i in range(n + 1):
        l.append(0)
        fx.append(1)
    l.append(n + 1)
    m = n
    fx.append(bin_to_int(xor(int_to_bin(fx[-1]), exponent_to_bin(n + 1))))

    print("n(0) = " + str(n))
    for i in range(n + 1):
        print("C" + str(n) + "(x) = ", end="")
    print("1")
    for i in range(n + 1):
        print("l" + str(n) + " = ", end="")
    print(0)
    print("n = " + str(n + 1) + "   ", end="")
    print()

    n += 1
    
    print("l(" + str(n) + ")=" + str(l[-1]) + "    ", end="")
    print_int_polynome(int_to_bin(fx[-1]), with_fx=n)
    print()
    
    print("r  |  дельта  |  L  |  C(x)")
    print("___________________________")
    print()
    while n < len(flow):
        d = bin_sum(xor(flow[:n + 1], int_to_bin(fx[-1])[::-1], 'right&'))
        if d == 0:
            fx.append(fx[-1])
            l.append(l[-1])
        else:
            '''
            f(n+1)(x) = fn(x)+dn dm^-1 x^(n-m) fm(x)
            '''
            fx.append(bin_to_int(xor(int_to_bin(fx[-1]), polynome_multiply(exponent_to_bin(n - m), int_to_bin(fx[m])))))

            if l[-1] >= n + 1 - l[-1]:
                l.append(l[-1])
            else:
                l.append(n + 1 - l[-1])
                m = n
            #print_int_polynome(int_to_bin(fx[-1]), with_fx=n + 1)
        if n <= 9:
          print(f"{n + 1}  |     {d}    |  {l[-1]}", end ="  |  ")
        else:
          print(f"{n + 1} |     {d}    |  {l[-1]}", end ="  |  ")
        # print("n = " + str(n) + "   ", end="")
        # print("d(" + str(n) + ") = " + str(d) + "  ", end="")
        # print("m = " + str(m) + "  ")
        # print("l(" + str(n) + ")=" + str(l[-1]) + "    ", end="")
        print_int_polynome(int_to_bin(fx[-1]), with_fx=n + 1)
        print()

        n += 1
    return [fx[-1], l[-1]]


def bin_sum(_bin):
    '''
    bin_sum([1,1]) = 0
    bin_sum([1,1,1]) = 1
    '''
    out = 0
    for i in _bin:
        out = out ^ i
    return out


def int_to_bin(_int):
    '''
    int_to_bin(15) = [1, 1, 1, 1]
    '''
    return [int(i) for i in list(bin(_int)[2:])]


def int_xor(_int1, _int2):
    return xor(int_to_bin(_int1), int_to_bin(_int2))


def bin_to_int(_bin):
    '''
    bin_to_int(['1', '1', '1', '1']) = 15
    '''
    out = ""
    for i in _bin:
        out += str(i)
    return int(out, 2)


def xor(_bin1, _bin2, mode='left'):
    '''
    xor([1, 0, 0, 0, 1, 0], [1, 0, 1, 1, 0],"right")=[1, 1, 0, 1, 0, 0]
    '''
    if len(_bin1) < len(_bin2):
        _bin1, _bin2 = _bin2, _bin1
    if mode == 'left':
        return [_bin1[i] ^ _bin2[i] for i in range(len(_bin2))] + _bin1[len(_bin2):]
    elif mode == 'right':
        return _bin1[:len(_bin1) - len(_bin2)] + [_bin1[len(_bin1) - len(_bin2) + i] ^ _bin2[i] for i in range(len(_bin2))]
    elif mode == "right&":
        return [0 for i in range(len(_bin1) - len(_bin2))] + [_bin1[len(_bin1) - len(_bin2) + i] & _bin2[i] for i in range(len(_bin2))]
    elif mode == "left&":
        return [_bin1[i] & _bin2[i] for i in range(len(_bin2))] + [0 for i in range(len(_bin1) - len(_bin2))]
    else:
        raise AttributeError
    # for j in range(len(_bin2))


def exponent_to_bin(exponent):
    '''
    x^2 => [0,0,1]
    exponent_to_bin(2) = [0,0,1]
    '''
    return [0 for i in range(exponent)] + [1]


def polynome_multiply(polynome1, polynome2):
    '''
    (x + x^2) * (x^2 + x^3 + x^6) = x^3 + x^5 + x^7 + x^8
    polynome_multiply([0,1,1],[0,0,1,1,0,0,1,0])=[0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    '''
    out = []
    for i in range(len(polynome1)):
        if polynome1[i] == 1:
            out = xor(out, [0 for j in range(i)] + polynome2)
    return out


def print_int_polynome(_bin, **arg):
    '''
    f(255) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7
    '''
    start = False
    if "with_fx" in arg and arg["with_fx"] == 0:
        print("---------------------------")
        print("Получилось C(x) =", end="")
    elif "with_fx" in arg and arg["with_fx"] != 0:
        print("C" + str(arg["with_fx"]) + "(x) =", end="")
    if _bin[0] == 1:
        print(" 1", end="")
        start = True
    if _bin[1] == 1:
        if start: print(" + x", end="")
        else:
            print(" x", end="")
            start = True
    for i in range(len(_bin) - 2):
        if _bin[i + 2] == 1:
            if start: print(" + x^" + str(i + 2), end="")
            else:
                print(" x^" + str(i + 2), end="")
                start = True
    print()

def test_bm():
    flow1 = list(map(int, input("Введите последовательность битов: ").split()))
    [fx, l] = simple_bm(flow1, debug=True)
    print_int_polynome(int_to_bin(fx), with_fx=0)
    print(f"L = {l}")

def calc_key(key, g, p):
    return (g ** key) % p


def DIFFIE_HELLMAN():

    number_of_users = int(input("Введите количество человек: "))
    
    if number_of_users == 2:
      A = int(input("Введите загаданное число для A: "))
      B = int(input("Введите загаданное число для B: "))
      g = int(input("Введите простое g(g<n): "))
      p = int(input("Введите простое n: "))

      arr_of_check = []
      for i in range (p):
        temp_var = mod(str(int(pow(g,i))), p)
        arr_of_check.append(temp_var)
      count = 0
      for digit in range(len(arr_of_check)):
          if digit in arr_of_check:
              count+=1
      print(arr_of_check)
      if count == p-1:
        pass
      else:
        print("Выберите другие значения, g не подходит")
        return

      A_public_key = calc_key(A, g, p)
      B_public_key = calc_key(B, g, p)

      A_shared_key = calc_key(A, B_public_key, p)
      B_shared_key = calc_key(B, A_public_key, p)
      print('Общий секретный ключ: {}'.format(A_shared_key))
      pass
    elif number_of_users == 3:
      A = int(input("Введите загаданное число для A: "))
      B = int(input("Введите загаданное число для B: "))
      C = int(input("Введите загаданное число для C: "))
      g = int(input("Введите g: "))
      p = int(input("Введите n: "))

      arr_of_check = []
      for i in range (p):
        temp_var = mod(str(int(pow(g,i))), p)
        arr_of_check.append(temp_var)
      count = 0
      for digit in range(len(arr_of_check)):
          if digit in arr_of_check:
              count+=1
      print(arr_of_check)
      if count == p-1:
        pass
      else:
        print("Выберите другие значения, g не подходит")
        return
      
      print("Шаг        |                Кто какой информацией будет обладать по шагам                |")
      print("-------------------------------------------------------------------------------------------")
      print("           |           A             |              B            |         C             |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 1      |    g={g} ;  n = {p}        |  g={g} ;  n = {p}            |  g={g} ;  n = {p}        | ")
      print("-------------------------------------------------------------------------------------------")

      A_public_key = calc_key(A, g, p)
      B_public_key = calc_key(B, g, p)
      C_public_key = calc_key(C, g, p)
      print(f"Шаг 2      |    n,g ; x = {A}; X = {A_public_key}  |        n,g ; X = {A_public_key}.      |      n,g              |")
      print("-------------------------------------------------------------------------------------------")

      A_calc_key = calc_key(A, C_public_key, p)
      B_calc_key = calc_key(B, A_public_key, p)
      C_calc_key = calc_key(C, B_public_key, p)

      print(f"Шаг 3      |        n,g ; x; X       |  n,g ; X ; y = {B}; Y = {B_public_key}  | n,g ; Y = {B_public_key}           |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 4      |     n,g ; x; X; Z={C_public_key}     |     n,g ; X ; y ; Y       |     n,g ; Y; Z={C_public_key}      |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 5      |  n,g ; x; X; Z; Z' = {C_calc_key}  |  n,g ; X; y; Y; Z' = {C_calc_key}    |   n,g ; Y; Z={C_public_key}        |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 6      |  n,g ; x; X; Z; Z' = {C_calc_key}  |n,g ; X; y; Y; Z' ; X' = {B_calc_key}| n,g ; Y; Z; X'={B_calc_key}     |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 7      | n,g ; x; X; Z; Z'; Y'={C_calc_key} |n,g ; X; y; Y; Z' ; X' = {B_calc_key}|n,g; Y; Z; X'; Y'={C_calc_key}    |")
      print("-------------------------------------------------------------------------------------------")

      A_shared_key = calc_key(A, C_calc_key, p)
      B_shared_key = calc_key(B, A_calc_key, p)
      C_shared_key = calc_key(C, B_calc_key, p)

      print(f"Шаг 8      | k1={A_shared_key}, проверка k1 = {calc_key(1,pow(pow(pow(g,B),C),A),p)} |                           |                       |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 9      |                         | k2={B_shared_key}, проверка k2 = {calc_key(1,pow(pow(pow(g,C),A),B),p)}   |                       |")
      print("-------------------------------------------------------------------------------------------")
      print(f"Шаг 10     |                         |                           |k3={C_shared_key}, проверка k3 = {calc_key(1,pow(pow(pow(g,A),B),C),p)}|")


      print('Общий секретный ключ: {}'.format(A_shared_key))
      pass
    elif number_of_users == 4:
      A = int(input("Введите загаданное число для A: "))
      B = int(input("Введите загаданное число для B: "))
      C = int(input("Введите загаданное число для C: "))
      D = int(input("Введите загаданное число для D: "))
      
      g = int(input("Введите g: "))
      n = int(input("Введите n: "))
      print()
      
      arr_of_check = []
      for i in range (n):
        temp_var = mod(str(int(pow(g,i))), n)
        arr_of_check.append(temp_var)
      count = 0
      for digit in range(len(arr_of_check)):
          if digit in arr_of_check:
              count+=1
      print(arr_of_check)
      if count == n-1:
        pass
      else:
        print("Выберите другие значения, g не подходит")
        return
      
      X_public_key = mod(str(int(pow(g,A))),n)
      Y_public_key = mod(str(int(pow(g,B))),n)
      Z_public_key = mod(str(int(pow(g,C))),n)
      W_public_key = mod(str(int(pow(g,D))),n)

      W_half_calc_key = mod(str(int(pow(W_public_key,A))),n)
      X_half_calc_key = mod(str(int(pow(X_public_key,B))),n)
      Y_half_calc_key = mod(str(int(pow(Y_public_key,C))),n)
      Z_half_calc_key = mod(str(int(pow(Z_public_key,D))),n)

      A_shared_key = mod(str(int(pow(Z_half_calc_key,A))),n)
      B_shared_key = mod(str(int(pow(W_half_calc_key,B))),n)
      C_shared_key = mod(str(int(pow(X_half_calc_key,C))),n)
      D_shared_key = mod(str(int(pow(Y_half_calc_key,D))),n)

      A_result_key = mod(str(int(pow(D_shared_key, A))), n)
      B_result_key = mod(str(int(pow(A_shared_key, B))), n)
      C_result_key = mod(str(int(pow(B_shared_key, C))), n)
      D_result_key = mod(str(int(pow(C_shared_key, D))), n)

      print(f"Ключ А = {A_result_key}, B = {B_result_key}, C = {C_result_key}, D = {D_result_key}")
    
def MTI():
  n = int(input("Введите простое n: "))
  g = int(input("Введите простое g(<n): "))

  A = int(input("Введите a(<n-2): "))
  B = int(input("Введите b(<n-2): "))
  
  Za_calc = calc_key(A, g, n)
  Zb_calc = calc_key(B, g, n)

  x = int(input("Введите x(<n-2): "))
  X = calc_key(x, g, n)

  y = int(input("Введите y(<n-2): "))
  Y = calc_key(y, g, n)
  A_calc_result = calc_key(1, pow(Y, A) * pow(Zb_calc, x), n)
  B_calc_result = calc_key(1, pow(X, B)* pow(Za_calc,y), n)
  print(f"k = {A_calc_result}, проверка, вычислим k(с чертой) = {B_calc_result}")

def ESP_DSA():
  p = int(input("Введите простое число p(например 23): "))
  q = int(input("Введите простое число q(например 11): "))
  t = int(input("Введите число t(<p): "))
  if (p-1) % q == 0:
    print("q делит p-1")
  else:
    print("Выберите другие числа p и q")
    return 
  if mod(str(int(pow(t,((p-1)/q)))), p) == 1:
    print("Выберите другое число t - это не подходит")
    return 
  else:
    h_m = int(input("Введите хэш значение h(m): "))
    g = mod(str(int(pow(t,(p-1)/q))), p)
    x = int(input("Введите число x(<q например 2): "))
    print("Найдем y")
    y = mod(str(pow(g,x)),p)
    print(f"Открытый ключ (p, q, g, y) = ({p}, {q}, {g}, {y}), закрытый ключ x = {x}")
    print("Вычислим цифровую подпись для сообщения h(m)")
    k = int(input("Введите k(<q например 4): "))
    smth, k_minus_one, smth_2 = extended_gcd(k,q)
    if k_minus_one < 0:
      k_minus_one = -(k_minus_one)%q
    r = mod(str(mod(str(pow(g, k)),p)),q)
    s = mod(str(k_minus_one * (x*r + h_m)), q)
    if  0< r < q and 0 < s < q:
      print(f'Результат формирования (r, s) = ({r},{s})')
      print()
    else:
      print("Следует заного сгенерировать подпись, поменять например k.")
      return 
    print("Выполним проверку")
    smthfb3,s_minus_one, sthfj_2 = extended_gcd(s, q)
    if s_minus_one < 0:
      s_minus_one = -(s_minus_one)%q
    v = mod(str(s_minus_one),q)
    z1 = mod(str(h_m*v),q)
    z2 = mod(str(r*v),q)
    u = mod(str(mod(str(pow(g,z1)*pow(y,z2)),p)),q)
    if r == u:
      print(f"Подпись подлинная u = {u}")
    else:
      print("Где-то ошибка")

def ESP_EL_GAMAL():
  print("p выбирается таким, чтобы выполнялось равенство p = 2q + 1,где q - также простое число. Тогда в качестве g можно взять любое число")
  print()
  p = int(input("Введите простое нечетное число p: "))
  g = int(input("Введите число g(<p например 2): "))
  print()
  if mod(str((int(pow(g,41)))),p) != 1:
    print("g подходит")
  arr_of_check = []
  for i in range (p):
    temp_var = mod(str(int(pow(g,i))), p)
    arr_of_check.append(temp_var)
  count = 0
  for digit in range(len(arr_of_check)):
    if digit in arr_of_check:
      count+=1
  print(arr_of_check)
  if count == p-1:
    x = int(input("Введите число x (<p-1): "))
    y = mod(str(int(pow(g,x))),p)
    h_m=int(input("Введите хэш-значение h(m): "))
    k = int(input("Введите k(простое с p-1): "))
    if gcd(k,p-1):
      hz_val,k_minus_one,hz_val2 = extended_gcd(k,p-1)
      if k_minus_one <0:
        k_minus_one = -(k_minus_one)%(p-1)
      a = mod(str(int(pow(g,k))),p)
      b = mod(str(int(-(h_m - x*a)*k_minus_one)),p-1)
      print(f"Получилась подпись: ({a},{b})")
      print()
      print("Проверим подпись")
      validate_1 = mod(str(int(pow(y,a))*int(pow(a,b))),p)
      print(f"Значениe y^a*a^b = {validate_1}")
      validate_2 = mod(str(int(pow(g,h_m))),p)
      print(f"Значениe g^h(m) mod p = {validate_1}")
      if validate_1 == validate_2:
        print("Подпись подлинная")
  else:
    print("Попробуйте заного выбрать число g, это не подходит")


def ESP_GOST():
  p_param = int(input("Введите простое число p: "))
  q_param = int(input("Введите простое число q(p-1 должно делиться на q): "))
  if (p_param-1)%q_param ==0:
    g_param = int(input("Введите целое число g(такое, что g^q mod p = 1; <p-1): "))
    if mod(str(int(pow(g_param,q_param))), p_param) == 1:
      print("Значения проходят проверку")
      print()
      x = int(input("Введите целое число x: "))
      y = mod(str(int(pow(g_param,x))),p_param)
      print(f"Получили открытый ключ (p, q, g, y) = ({p_param}, {q_param}, {g_param}, {y}) и закрытый ключ x = {x}")
      print()
      print("Сформируем подпись")
      hash_m = int(input("Введите хэш-значение h(m): "))
      k = int(input("Введите число k(<q): "))
      temp_val_for_r= mod(str(int(pow(g_param,k))),p_param)
      r = mod(str(temp_val_for_r),q_param)
      s = mod(str(int((x*r + k*hash_m))),q_param)
      if r*s ==0:
        print("Выберите другие значения, т.к. r*s = 0")
        return
      else:
        print("Все норм")
        print(f"Цифровая подпись равна r mod2^256 = {mod(str(r),int(pow(2,256)))} и s mod2^256 = {mod(str(s), int(pow(2,256)))}")
        print()
        print("Проверим подпись")
        z0 = mod(str(int((pow(hash_m,q_param-2)))), q_param)
        z1 = mod(str(s*z0),q_param)
        z2 = mod(str(int((q_param-r)*z0)),q_param)
        temp_u = mod(str(int(pow(g_param,z1)*pow(y,z2))),p_param)
        u = mod(str(temp_u),q_param)
        print(f"Получились значения u = {u} и r = {r}")
        if u == r:
          print("Подпись подлинная, а сообщение неизменно")
    else:
      print("Выберите другое g")
      return
  else:
    print("Введите другие значения чисел p и q")
    return
  
    
  
  


if __name__=="__main__":    
  print("Общая программа для всего:")
  print()
  print("1 - (LFSR?)Построить регистр сдвига с линейной обратной связью с ассоцированным многочленом ... и выписать состояние регистра, если он был инициализирован вектором ...")
  print()
  print("2 - Найти модуль числа")
  print()
  print("3 - Расширенный алгоритм Евклида(Поиск модуля 3^-1 = 1 mod 5 на вход (3, 5), ответ x)")
  print()
  print("4 - Задача про рюкзаки, где дан вектор B и найти исходный вектор А")
  print()
  print("5 - Найти 4-х битовый регистр сдвига, если известна последовательность битов, которая была сгенерирована")
  print()
  print("6 - Построить 10-битную псевдослучайную последовательность с помощью BBS генератора(Блюм)")
  print()
  print("7 - Построить 10-битную псевдослучайную последовательность с помощью RSA генератора")
  print()
  print("8 - Построить аддитивный генератор, ассоциированный с многочленом ...(x^4+x^3+1) и n = 4. Начальное состояние генератора - массив ...(12,2,0,11)")
  print()
  print("9 - (Алгоритм Берлекэмпа-Месси)Найдем регистр сдвига с линейной обратной связью, если нам известна последовательность битов, которая была им сгенерирована ... (01011110001)")
  print()
  print("10 - Алгоритм DIFFIE-HELLMAN")
  print()
  print("11 - Алгоритм MTI")
  print()
  print("12 - Алгоритм ЭЦП DSA")
  print()
  print("13 - Алгоритм ЭЦП Эль-Гамаля")
  print()
  print("14 - Алгоритм ЭЦП Гост")


  condition = int(input("Что посчитать? Введите цифру: "))

  # only for two variables
  if condition == 1:
    parameters_of_polynomial = list(map(int, input("Последовательность степеней ассоц. многочлена(для 2-х чисел в убывающ. порядке): ").split()))
    init_state = list(map(int, input("Последовательность init_state: ").split()))
    print(init_state)
    print(parameters_of_polynomial)
    calc_output_stream(init_state, pow(2,len(init_state))-1,parameters_of_polynomial)
    
  elif condition == 2:
    positive_num = input("Положительное число? + Да, - Нет ")
    if positive_num == "+":
      num = pow(49,8)
      num2 = pow(19,2)
      multip = num*num2
      num = str(num)
      multip = str(multip)
      print(mod(str(3), 5))
    else:
      # -число % модуль
      print(-(844)%713)
      
  elif condition == 3:
    # extended_gcd( первое число, второе число)
    gcd, x, y = extended_gcd(25, 41)
    print('The GCD is', gcd)
    print(f'x = {x}, y = {y}')
    
  elif condition == 4:
    summa = 0
    from fractions import gcd
    arr = [1,2,3]
    arr2 = [1, 2, 3]
    symbolK = "k"
    B = list(map(int, input().split()))
    print(f"Введен вектор B = {B}")
    backpack_find_A(B, arr, arr2, summa, symbolK)
    
  elif condition == 5:
    sequence = list(map(int,input("Введите последовательность: ").split()))
    s4 = {"c1": sequence[3], "c2":sequence[2], "c3":sequence[1], "c4":sequence[0]}
    for num in list(s4):
      if s4[num] !=1:
        del s4[num]
    print(s4,"  ", sequence[4])
    s5 = {"c1": sequence[4], "c2":sequence[3], "c3":sequence[2], "c4":sequence[1]}
    for num in list(s5):
      if s5[num] !=1:
        del s5[num]
    print(s5,"  ", sequence[5])
    s6 = {"c1": sequence[5], "c2":sequence[4], "c3":sequence[3], "c4":sequence[2]}
    for num in list(s6):
      if s6[num] !=1:
        del s6[num]
    print(s6,"  ", sequence[6])
    s7 = {"c1": sequence[6], "c2":sequence[5], "c3":sequence[4], "c4":sequence[3]}
    for num in list(s7):
      if s7[num] !=1:
        del s7[num]
    print(s7, "  ", sequence[7])
    
  elif condition == 6:
    BBS()
    
  elif condition == 7:
    RSA()
    
  elif condition == 8:
    polynomial = list(map(int,input("Введите многочлен(степени): ").split()))
    initial_state = list(map(int,input("Введите начальное состояние ").split()))
    start_poson = []
    for i in initial_state:
      start_poson.append(i)
    n = int(input("Введите число n: "))
    arr = []
    print("Такт  |  Выход  |  Заполнение регистров  |")
    for i in range(16): # не понятно до какого такта заполнять табл.
      res = generate_next_numb(initial_state[polynomial[0]-1]+initial_state[polynomial[1]-1],2**n,1)
      initial_state.pop(-1)
      initial_state.insert(0,res)
      arr.append(res)
      print(f"{i+1}  |  {res}  |  {initial_state}  ")
    print(f"На выходе аддитивного генератора получили последовательность: {arr}")
    print("--------------")
    print("Такт  |  Выход  |  Заполнение регистров  |")
    # key_table = [2, 1, 0, 12, 5, 3, 13, 10, 5, 14, 8, 7, 11, 9, 15, 4] # непонятно откуда взять эту таблицу(скорее всего дана в условии)
    #key_table = [2, 1, 0, 12, 5, 3, 13, 10, 5, 14, 8, 7, 11, 9, 15, 4]
    key_table = [9, 10, 3, 2, 15, 4, 28, 6, 24, 1, 0, 11, 26, 18, 14, 23, 31, 8, 30, 19, 5, 13, 22, 7, 17, 25, 20, 27, 12, 29, 21, 16]
    arr_stoch_seq =[]
    for i in range(12):
      for j in range(32): # размерность n
        if start_poson[polynomial[1]-1] == key_table[j]:
          out_res = key_table[generate_next_numb(j + start_poson[polynomial[0]-1],pow(2,n),1)]
          break
      start_poson.pop(-1)
      start_poson.insert(0, out_res)
      arr_stoch_seq.append(out_res)
      print(f"{i+1}  |  {out_res}  |  {start_poson}  ")
    print(f"На выходе стохастического генератора получили последовательность {arr_stoch_seq}")
    
  elif condition == 9:
    test_bm()
    
  elif condition == 10:
    DIFFIE_HELLMAN()
    
  elif condition == 11:
    MTI()
    
  elif condition == 12:
    ESP_DSA()
    
  elif condition == 13:
    ESP_EL_GAMAL()
  elif condition == 14:
    ESP_GOST()

