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
    for i in range(10): # количество бит(длина в текущем примере равна 10-битной)
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
    # seryjnyTest(outputArray)
    # pokerowyTest(outputArray)
    # seriiTest(outputArray, '0')
    # seriiTest(outputArray, '1')


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
  q = randomprime(length = 6 // 2, millerrounds = 2)
  n = p * q
  phi = (p-1)*(q-1)
  e = 0
  bitArray = []
  outputArray = []
  for i in range(40):
    if gcd(phi, i) == 1:
      e = i
  e = 3
  start_x = 14 # стартовое число генератора x0
  if start_x % n:
    outputArray.append(0)
  else:
    outputArray.append(1)
  howManyZero = 0
  howManyOne = 0
  for i in range(9): # количество бит(длина в текущем примере равна 10-битной)
        start_x = generate_next_numb(start_x, n, e)
        bitArray = list('{0:0b}'.format(start_x))
        print(f"Число {i}: {start_x},  младший бит {bitArray[len(bitArray) - 1]}")
        if bitArray[len(bitArray) - 1] == '0':
            howManyZero += 1
        if bitArray[len(bitArray) - 1] == '1':
            howManyOne += 1
        outputArray.append(bitArray[len(bitArray) - 1])
  print (outputArray)
  print(f"p = {p}")
  print(f"q = {q}")
  print(f"n = {n}")
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



if __name__=="__main__":	
  print("Общая программа для всего:")
  print()
  print("1 - (LFSR)Построить регистр сдвига с линейной обратной связью с ассоцированным многочленом ... и выписать состояние регистра, если он был инициализирован вектором ...")
  print()
  print("2 - Найти модуль числа")
  print()
  print("3 - Расширенный алгоритм Евклида")
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
      print(mod(str(15), 16))
    else:
      # -число % модуль
      print(-(844)%713)
  elif condition == 3:
    # extended_gcd( первое число, второе число)
    gcd, x, y = extended_gcd(68, 315)
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
    key_table = [2, 1, 0, 12, 5, 3, 13, 10, 5, 14, 8, 7, 11, 9, 15, 4] # непонятно откуда взять эту таблицу(скорее всего дана в условии)
    arr_stoch_seq =[]
    for i in range(8):
      for j in range(16):
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
