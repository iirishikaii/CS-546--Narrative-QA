from itertools import zip_longest
li = [1,2,3,4,5]

args = [iter(li)] * 3
out = zip_longest(*args)

print (list(out))


li1 = ["2.4","3.6","1.2"]
li1 = ["dhruv","agarwal","dxero"]
max1=li1[0]
for i in li1:
    if i>max1:
        max1=i

print(max1)
print(max(li1))

li2=['a','b']
i=0
sum=0
while  i <len(li2) and li2[i] != -1 :
    sum+=ord(li2[i])
    i+=1

print(sum)

st='a'
if st >= '0' and st <='9':
    print ("you are right on")
