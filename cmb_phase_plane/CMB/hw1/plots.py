import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

"""
t = [0,10,10.00001,20,20.00001,30]
minV = 10
maxV = 110
V = [minV,minV,maxV,maxV,minV,minV]

plt.plot(t,V)
plt.plot([0,30],[0,0],'black',ls='dashed')
plt.ylim([minV-15,maxV+15])
plt.xlabel('t (msec)')
plt.ylabel('V (mV)')
plt.title('Potassium Driving Force')
plt.show()
"""

def n(t):
    return 1 - np.exp(-t/5.0)

def m(t):
    return 1 - np.exp(-2.0*t)

def h(t):
    return np.exp(-t/5.0)

def n2(t):
    return (1 - np.exp(-2))*np.exp(-t/5.0)

def m2(t):
    return (1 - np.exp(-20))*np.exp(-2.0*t)

def h2(t):
    return (np.exp(-2) - 1)*np.exp(-t/5.0) + 1


t1 = [0,10]
t2 = np.arange(10,20,0.01)
t3 = np.arange(20,30,0.01)

p1, p2, p3 = plt.plot(t1,[0.001,0.001],'red',t1,[-0.001,-0.001],'green',t1,[1,1],'blue')
plt.plot(t2,n(t2-10),'red',t2,m(t2-10),'green',t2,h(t2-10),'blue')
plt.plot(t3,n2(t3-20),'red',t3,m2(t3-20),'green',t3,h2(t3-20),'blue')
plt.legend([p1, p2, p3], ["n","m","h"])

plt.ylim([-0.1,1.1])
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Gating Variables")
plt.show()

def gna(t):
    return 120*m(t)**3*h(t)

def gk(t):
    return 36*n(t)**4

def gna2(t):
    return 120*m2(t)**3*h2(t)

def gk2(t):
    return 36*n2(t)**4

p1,p2 = plt.plot(t1,[0.02,0.02],'purple',t1,[-0.02,-0.02],'orange')
plt.plot(t2,gna(t2-10),'purple',t2,gk(t2-10),'orange')
plt.plot(t3,gna2(t3-20),'purple',t3,gk2(t3-20),'orange')
plt.legend([p1,p2],["g_Na", "g_K"])
plt.xlabel("Time")
plt.ylabel("Conductance")
plt.title("Ion Channel Conductances")
#plt.ylim([-0.2,3.5])
plt.show()

def Ina(t):
    return -10*gna(t)

def Ik(t):
    return 110*gk(t)

def Ina2(t):
    return -10*gna2(t)

def Ik2(t):
    return 110*gk2(t)



p1,p2 = plt.plot(t1,[0.5,0.5],'purple',t1,[-0.5,-0.5],'orange')
plt.plot(t2,Ina(t2-10),'purple',t2,Ik(t2-10),'orange')
plt.plot(t3,Ina2(t3-20),'purple',t3,Ik2(t3-20),'orange')
plt.legend([p1,p2],["I_Na", "I_K"])
plt.xlabel("Time")
plt.ylabel("Current")
plt.title("Ionic Current Across Membrane")
plt.show()
