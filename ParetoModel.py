from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:50:01 2016

@author: Trent
"""

import Tkinter as tk
import tkFileDialog
from scipy.stats import genpareto
import numpy
import matplotlib.pyplot as plt
import math
import csv


#File Selection GUI
print 'Please select a CSV File'
print 'File may have multiple columns'

root1 = tk.Tk()

fileName = tkFileDialog.askopenfilename(parent=root1,title='Choose a CSV File',filetypes={('Comma Separated Value Files', '.csv'),('all files','*')})
tk.Label(root1, text='Click Run Program to Input Data').grid(row=0)
tk.Label(root1, text='Select either a CSV file or a Tab Delimited Text File').grid(row=0)
tk.Button(root1, text='Run Program', height=2, width=10, command=root1.destroy).grid(sticky=tk.W)
root1.mainloop()

print 'File Being Used: ' +str(fileName)
print '\n'
data = []
testdata = []
linesofdata = []


#Importing text Files
#if fileName != '' and fileName[len(fileName)-3]+fileName[len(fileName)-2]+fileName[len(fileName)-1]=='txt':
#    print 'Please enter the Column number you wish to import, Click "Import Column Number" then click "Continue"'
#    tcolumnlist = []
#    troot = tk.Tk()
#    def getcolumntext():
#        print("Column Number: %s" % tentry1.get())
#        print("Data from Column Number %s will be used" % tentry1.get())
#        print '\n'
#        tcolumnlist.append(int(tentry1.get()))
#    troot.wm_title('Input the Column from the Text File you wish to use, click "Import Column Number" then click "Continue"')
#    tk.Label(troot, text = "Select which Column from the Text File to Import").grid(row=0)
#    tk.Label(troot, text = "Integer Values Only").grid(row=0,column=2)
#    tentry1 = tk.Entry(troot)
#    tentry1.grid(row=0,column=1)
#    tentry1.insert(10,"1")
#    tk.Button(troot, text='Import Column Number', command=getcolumntext).grid(row=2, column=1, sticky=tk.W, pady=4)
#    tk.Button(troot, text='Continue', command=troot.destroy).grid(row=2, column=0, sticky=tk.W, pady=4, padx=4)
#    troot.mainloop()
#    #Parse the file data - Tab Delimited
#    actualfile = open(fileName,"r+")
#    data = actualfile.read()
#    data = data.split('\n')
#    del data[-1]
#    for lines in data:
#        linesofdata.append(lines.split('\t'))
#    for entries in linesofdata:
#        if entries[int(tcolumnlist[len(tcolumnlist)-1])-1] != '':
#            try:
#                testdata.append(float(entries[int(tcolumnlist[len(tcolumnlist)-1])-1]))
#            except ValueError:
#                pass

#Importing csv Files
if fileName != '' and fileName[len(fileName)-3]+fileName[len(fileName)-2]+fileName[len(fileName)-1]=='csv' or fileName != '' and fileName[len(fileName)-3]+fileName[len(fileName)-2]+fileName[len(fileName)-1]=='CSV':
    print 'Please enter the Column number you wish to import, Click "Import Column Number" then click "Continue"'
    columnlist = []
    #Parse the file data - CSV File
    root3 = tk.Tk()
    def getcolumn():
        print("Column Number: %s" % entry1.get())
        print("Data from Column Number %s will be used" % entry1.get())
        print '\n'
        columnlist.append(int(entry1.get()))
    root3.wm_title('Input the Column from the CSV you wish to use, click "Import Column Number" then click "Continue"')
    tk.Label(root3, text = "Select which Column from the CSV File to Import").grid(row=0)
    tk.Label(root3, text = "Integer Values Only").grid(row=0,column=2)
    entry1 = tk.Entry(root3)
    entry1.grid(row=0,column=1)
    entry1.insert(10,"1")
    tk.Button(root3, text='Import Column Number', command=getcolumn).grid(row=2, column=1, sticky=tk.W, pady=4)
    tk.Button(root3, text='Continue', command=root3.destroy).grid(row=2, column=0, sticky=tk.W, pady=4, padx=4)
    root3.mainloop()
    with open(fileName, "rb") as f:
        reader = csv.reader(f)
        testdata = []
        for row in reader:
            if row!=[] and len(row)>=columnlist[-1] and row[columnlist[-1]-1]!='':
                try:
                    testdata.append(float(row[columnlist[len(columnlist)-1]-1]))
                except ValueError:
                    pass

#Fake file... Just used to test the model against random genpareto data
#Take this whole thing out when I give it to them...
#This is purely Hypothetical...
if fileName == '':
    print 'Please restart the application and select a CSV File'
    print '\n'
    root5 = tk.Tk()
    fakedata = []
    def getfakedata():
        print("Guess Shape: %s\nGuess Threshold: %s\nGuess Scale: %s" % (eentry1.get(), eentry2.get(), eentry3.get()))
        fakedata.append(float(eentry1.get()))
        fakedata.append(float(eentry2.get()))
        fakedata.append(float(eentry3.get()))
        
    root5.wm_title('Select Parameter Values to generate "random" Gen Pareto Data for the model to be trained with')
    tk.Label(root5, text = "Guess Shape").grid(row=0)
    tk.Label(root5, text = "(-1, infinity)").grid(row=0, column=2)
    tk.Label(root5, text = "Guess Threshold").grid(row=1)
    tk.Label(root5, text = ">0").grid(row=1, column=2)
    tk.Label(root5, text = "Guess Scale").grid(row=2)
    tk.Label(root5, text = "(0, infinity)").grid(row=2, column=2)

    eentry1 = tk.Entry(root5)
    eentry2 = tk.Entry(root5)
    eentry3 = tk.Entry(root5)

    eentry1.grid(row=0,column=1)
    eentry1.insert(10,"1")
    eentry2.grid(row=1,column=1)
    eentry2.insert(10,"2")
    eentry3.grid(row=2,column=1)
    eentry3.insert(10,"2")
    tk.Button(root5, text='Import Data', command=getfakedata).grid(row=6, column=1, sticky=tk.W, pady=4)
    tk.Button(root5, text='Continue', command=root5.destroy).grid(row=6, column=0, sticky=tk.W, pady=4, padx=4)
    
    root5.mainloop()    
    
    testdata = [genpareto.rvs(fakedata[len(fakedata)-3], loc=fakedata[len(fakedata)-2], scale=fakedata[len(fakedata)-1]) for i in range(1000)]
    
testdata = sorted(testdata)

#Shifting Test data from uncalibrated readings... (Negative readings)
#if min(testdata)<0 and fileName != '':
#    utestdata = sorted(testdata)
#    testdata = []
#    mindatapoint = abs(min(utestdata))
#    for data in utestdata:
#        testdata.append(mindatapoint+data)

    
print 'Please enter values for Threshold Step Size, Threshold Min Value, and Threshold Max Value'
print 'Then click "Display Plot", when satisfied click "Continue" and note your selection of Threshold'
print '\n'
#Here we generate MRLP so user can select a threshold
root = tk.Tk()
print 'Please wait for the Program to compute certain values and to display the plot once you click "Display Plot"'
print 'The program will take longer to run with a lower threshold step size and a wider range between the minimum and maximum values for threshold'
print '\n'
print '*Remember to Note down the Threshold you wish to select from the graph. This value is needed in the next prompt*'
print '\n'

results = []
def MRLP():
    if float(e3.get())>=max(testdata):
        print 'Please select a smaller value for "Threshold Max Value", %s exceeds the maximum data value from the imported data file' % e3.get()
    if float(e3.get())<max(testdata):
        print("Threshold Step Size: %s\nThreshold Min Value: %s\nThreshold Max Value: %s" % (e1.get(), e2.get(), e3.get())) 
        print '\n'
        print 'If the graph pops up in a new window you must close out of the graph before displaying a new plot or continuing on with the program'
        print '\n'
        results.append((float(e1.get())))
        results.append((float(e2.get())))
        results.append((float(e3.get())))
        thresholdvalues = numpy.arange(float(e2.get()),float(e3.get())+1,float(e1.get())).tolist()
        global xvalues
        global yvalues
        global METlist
        global stdlist
        global thresholds
        global lengthlist
        global meanexcesses
        global MECI95list
        global upperCIME
        global lowerCIME
        xvalues = {}
        yvalues = {}
        METlist = []
        stdlist = []
        lengthlist = []
        for q in range(len(thresholdvalues)):
            xvalues['x'+str(q)] = []
            yvalues['y'+str(q)] = []
            for data in testdata:
                if data>thresholdvalues[q]:
                    xvalues['x'+str(q)].append(data)
                    yvalues['y'+str(q)].append(data-thresholdvalues[q])
            METlist.append((thresholdvalues[q],sum(yvalues['y'+str(q)])/len(yvalues['y'+str(q)])))
            stdlist.append(numpy.std(yvalues['y'+str(q)])) #Need the length of every yvalue list..... we have the std
            lengthlist.append(len(yvalues['y'+str(q)]))
        
        thresholds = [x[0] for x in METlist]
        meanexcesses = [y[1] for y in METlist]
        MECI95list = []
        for i in range(len(stdlist)):
            MECI95list.append(1.96*stdlist[i]/math.sqrt(lengthlist[i]))
        upperCIME = []
        for i in range(len(meanexcesses)):
            upperCIME.append(meanexcesses[i]+MECI95list[i])
        lowerCIME = []
        for i in range(len(meanexcesses)):
            lowerCIME.append(meanexcesses[i]-MECI95list[i])
        print 'Pick a Threshold Value where the graph appears stable and linear' 
        print 'Make sure the Threshold Value you pick falls in the range of the last plotted values'
        print 'Also make sure that the Threshold Value is a multiple of the step-size'
        print '\n'
        #Mean Residual Life Plot
        plt.title('Mean Residual Life Plot')
        plt.ylabel('Mean Excess')
        plt.xlabel('Threshold')
        plt.plot(thresholds,upperCIME, 'r--', label='Upper and Lower Confidence Intervals')
        plt.plot(thresholds,lowerCIME, 'r--')
        plt.plot(thresholds,meanexcesses, label = 'Mean Residual Life Plot')
        plt.legend(bbox_to_anchor=(0.,1.1,1.,.102), loc=3,ncol=1,mode="expand",borderaxespad=0.)
        plt.show()

        

root.wm_title('Input a Plot Range for the MRLP then click "Display Plot", Once satisfied click "Continue"')
tk.Label(root, text = "Threshold Step Size").grid(row=0)
tk.Label(root, text = "Integers Only").grid(row=0, column=2)
tk.Label(root, text = "Threshold Min Value").grid(row=1)
tk.Label(root, text = ">=0").grid(row=1, column=2)
tk.Label(root, text = "Threshold Max Value").grid(row=2)
tk.Label(root, text = "<Max Data Value").grid(row=2, column=2)
tk.Label(root, text = "Enter Values").grid(row=3)
tk.Label(root, text = "click 'Display Plot'").grid(row=4)
tk.Label(root, text = "then click 'Continue'").grid(row=5)

e1 = tk.Entry(root)
e2 = tk.Entry(root)
e3 = tk.Entry(root)

e1.grid(row=0,column=1)
e1.insert(10,"100")
e2.grid(row=1,column=1)
e2.insert(10,"0")
e3.grid(row=2,column=1)
e3.insert(10,"%s" % float(int(max(testdata))-1))

tk.Button(root, text='Display Plot', command=MRLP).grid(row=6, column=1, sticky=tk.W, pady=4)
tk.Button(root, text='Continue', command=root.destroy).grid(row=6, column=0, sticky=tk.W, pady=4, padx=4)


root.mainloop()
    
#print results
#print 'Threshold Step Size: ' + str(results[len(results)-3])
#print 'Threshold Min Value: ' + str(results[len(results)-2])
#print 'Threshold Max Value: ' + str(results[len(results)-1])
print '*Remember to select a threshold where the graph appears stable and linear*'
print 'If Threshold is stable and linear at multiple spots it is recommended to run through the program mutiple times selecting different thresholds every time and seeing how each model fits in the validation part of the program'
print '\n'
print 'Please Input Values for Threshold Value, Shape Step Size, Shape Min Guess Value, and Shape Max Guess Value'
print 'Then click "Store Values" followed by "Run Program"'
print '\n'


#User Input for Parameter Estimation (Range, stepsize, Threshold)
root2 = tk.Tk()

print 'The program will take longer to run with a lower shape step size and a wider range between the minimum and maximum guess values for shape'
print '\n'
results2 = []
def get_info2():
    if float(ee1.get()) not in thresholds:
        print(str('*ERROR: Threshold Value Selected: '+str(ee1.get())+' does not either fall within the Max/Min estimate for threshold or is not a multiple of the Threshold Step Size. Please select a different Threshold Value that follows the prior statement.' ))
        print '\n'
    if float(ee1.get()) in thresholds:
        print(str(ee1.get())+' is a valid Threshold Selection')
        print("Threshold Value: %s\nShape Step Size: %s\nShape Min Guess Value: %s\nShape Max Guess Value: %s" % (ee1.get(), ee2.get(), ee3.get(), ee4.get())) 
        print '\n'
        results2.append((float(ee1.get())))
        results2.append((float(ee2.get())))
        results2.append((float(ee3.get())))
        results2.append((float(ee4.get())))

        
root2.wm_title('Input Values, click "Store Values", then click "Continue"')
tk.Label(root2, text = "Threshold Value").grid(row=0)
tk.Label(root2, text = "Multiple of Threshold Step Size").grid(row=0, column=2)
tk.Label(root2, text = "Shape Step Size").grid(row=1)
tk.Label(root2, text = "(0,+infinity) usually a decimal (0,1)").grid(row=1, column=2)
tk.Label(root2, text = "Shape Min Guess Value").grid(row=2)
tk.Label(root2, text = "Choose 0 or -1").grid(row=2, column=2)
tk.Label(root2, text = "Shape Max Guess Value").grid(row=3)
tk.Label(root2, text = "(0,+infinity) Higher Value, Higher Time").grid(row=3, column=2)

ee1 = tk.Entry(root2)
ee2 = tk.Entry(root2)
ee3 = tk.Entry(root2)
ee4 = tk.Entry(root2)

ee1.grid(row=0,column=1)
ee2.grid(row=1,column=1)
ee2.insert(10,".1")
ee3.grid(row=2,column=1)
ee3.insert(10,"-1")
ee4.grid(row=3,column=1)
ee4.insert(10,"2")


tk.Button(root2, text='Store Values', command=get_info2).grid(row=4, column=1, sticky=tk.W, pady=4)
tk.Button(root2, text='Run Program', command=root2.destroy).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)

root2.mainloop()
    
#print results2
#print 'Threshold Value:' + str(results2[len(results2)-4])
#print 'Shape Step Size:' + str(results2[len(results2)-3])
#print 'Shape Min Guess Value:' + str(results2[len(results2)-2])
#print 'Shape Max Guess Value:' + str(results2[len(results2)-1])
#print '\n'  

#Interval stuff for 0, -1, and other min guess xis value...
if results2[len(results2)-2] !=0 and results2[len(results2)-2] !=-1:
    minrangexis = results2[len(results2)-2]
Threshold = results2[len(results2)-4]
xisstepsize = results2[len(results2)-3]
if results2[len(results2)-2]==0:
    minrangexis = results2[len(results2)-2]-results2[len(results2)-3]
if results2[len(results2)-2]==-1:
    minrangexis = results2[len(results2)-2]
maxrangexis = results2[len(results2)-1]+results2[len(results2)-3]

threshposition = thresholds.index(Threshold)
q=threshposition

#Derivative WRT Scale(sigma)
#Remember to implement the derivative when shape(xi) equals 0 and 1, special cases
        
def LL(sigmaa, xi):
    if xi != 0 and xi != -1:
        return -len(yvalues['y'+str(q)])*math.log(sigmaa) - (1 + 1/xi)*sum(math.log(yvalues['y'+str(q)][w]*xi/sigmaa + 1) for w in range(len(yvalues['y'+str(q)])))
    if xi == 0:
        return -len(yvalues['y'+str(q)])*math.log(sigmaa) - sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/sigmaa
    if xi == -1:
        return len(yvalues['y'+str(q)])*math.log(sigmaa)



print 'Number of data points that will train the model = '+str(len(yvalues['y'+str(q)]))
#Label For Optimization    
  
print str('Scale') + '          '  + str('Shape') + '  ' +str('Approximately Zero')
#Parameter Estimation for xis and sigma... Log Likelihood
from scipy.optimize import fsolve
import scipy.optimize
sigmalist = []
guessxis = [xisstepsize*x for x in range(int(minrangexis/xisstepsize+1),int(maxrangexis/xisstepsize))] #not sure what to guess when there is real data
for xis in guessxis:
    if xis>=0:
        def LLDsigma(sigma):
            if xis != 0:
                return -len(yvalues['y'+str(q)])/sigma - (1 + 1/xis)*sum(-yvalues['y'+str(q)][w]*xis/(sigma**2*(yvalues['y'+str(q)][w]*xis/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))
            if xis == 0:
                return -len(yvalues['y'+str(q)])/sigma + sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/sigma**2
        if xis == 0:
            solvedsigma = fsolve(LLDsigma, .00005)
            sigmalist.append(float(solvedsigma))
        if xis != 0:
            solvedsigma = fsolve(LLDsigma, .0005)
            sigmalist.append(float(solvedsigma))    
    if xis<0 and xis>-1:
        def LLDsigma(sigma):
            if xis != 0:
                return -len(yvalues['y'+str(q)])/sigma - (1 + 1/xis)*sum(-yvalues['y'+str(q)][w]*xis/(sigma**2*(yvalues['y'+str(q)][w]*xis/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))
            if xis == 0:
                return -len(yvalues['y'+str(q)])/sigma + sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/sigma**2
        try:
            solvedsigma = float(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)))
        except ZeroDivisionError:
            try:
                solvedsigma = float(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)+.1))
            except ValueError:
                try:
                    solvedsigma = float(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)+1))
                except ValueError:
                    pass
                    print 'Cannot find a solution for xis = '+str(xis)+' The algorithm will try the next guessed xis'
                    solvedsigma=None
        sigmalist.append(solvedsigma)
    if solvedsigma is not None:    
        print str(float(solvedsigma)) + '   '  + str(xis) + '   ' +str(LLDsigma(solvedsigma))

print str('Scale') + '          '  + str('Shape') + '  ' +str('Approximately Zero')
print '\n'
print 'Make sure the values under "Approximately Zero" are close to or equal to zero.'
print 'Values less than 1e-9 should suffice.'
print '\n'
print 'If you are not getting values close to zero under "Approximately Zero" then the algorithm is not learning the Scale parameter properly based on the corresponding Shape parameter.'
print 'If this problem is encountered start over with a different threshold selection or change the guess Shape range so that it does not include Shapes that learn improper Scale values (Where the "Approximately Zero" column is not close to zero).'

#Computing LLF values and plotting
LLlist = []
newxis = []
newsigma = []
for i in range(len(guessxis)):
    if sigmalist[i] is not None:
        if (1+(sigmalist[i]**-1)*guessxis[i]*max(yvalues['y'+str(threshposition)]))>0:
            LLlist.append(LL(sigmalist[i],guessxis[i]))
            newxis.append(guessxis[i])
            newsigma.append(sigmalist[i])

MaxLL = max(LLlist)
MaxLLindex = LLlist.index(max(LLlist))
Maxsigma = newsigma[MaxLLindex]
Maxxis = newxis[MaxLLindex]

print '\n'
print 'The Shape vs Log-Likelihood Plot should look similar to a bell shape curve and should display a maximum.'
print 'If there is not a definitive maximum on the plot please restart the program with a wider selection range on minimum and maximum shape values. However please do not select a minimum value of less than -1'
print '\n'
print '*If you would like a more accurate solution, restart the program with a more narrow selection range on the minimum and maximum shape values that surround the maximum. Also lower the shape step size in this interval and make sure to select the same threshold value.*'

#Shape vs Log-Likelihood Plot
plt.title('Shape vs Log-Likelihood')
plt.ylabel('Profile Log-Likelihood')
plt.xlabel('Shape Parameter')
plt.plot(newxis,LLlist)
plt.show()

#Confidence Intervals Need to implement when xi/shape=0...
def sigmadoubleprime(sigma, xi):
    if xi != 0:
        return len(yvalues['y'+str(q)])/sigma**2 - (1 + 1/xi)*sum(-yvalues['y'+str(q)][w]**2*xi**2/(sigma**4*(yvalues['y'+str(q)][w]*xi/sigma + 1)**2) + 2*yvalues['y'+str(q)][w]*xi/(sigma**3*(yvalues['y'+str(q)][w]*xi/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))
    if xi == float(0):
        return len(yvalues['y'+str(q)])/sigma**2 - 2*sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/sigma**3
def xidoubleprime(sigma, xi):
    if xi != 0:
        return (-1 - 1/xi)*sum(-yvalues['y'+str(q)][w]**2/(sigma**2*(yvalues['y'+str(q)][w]*xi/sigma + 1)**2) for w in range(len(yvalues['y'+str(q)]))) + 2*sum(yvalues['y'+str(q)][w]/(sigma*(yvalues['y'+str(q)][w]*xi/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))/xi**2 - 2*sum(math.log(yvalues['y'+str(q)][w]*xi/sigma + 1) for w in range(len(yvalues['y'+str(q)])))/xi**3
    if xi == float(0):
        return 0
def sigmaprimexiprime(sigma, xi):
    if xi != 0:
        return -(1 + 1/xi)*sum(yvalues['y'+str(q)][w]**2*xi/(sigma**3*(yvalues['y'+str(q)][w]*xi/sigma + 1)**2) - yvalues['y'+str(q)][w]/(sigma**2*(yvalues['y'+str(q)][w]*xi/sigma + 1)) for w in range(len(yvalues['y'+str(q)]))) + sum(-yvalues['y'+str(q)][w]*xi/(sigma**2*(yvalues['y'+str(q)][w]*xi/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))/xi**2
    if xi == float(0):
        return 0

#Make the Variance/Covariance Matrix and compute CI's
if Maxxis != 0:
    numpy.matrix([[sigmadoubleprime(Maxsigma, Maxxis),sigmaprimexiprime(Maxsigma, Maxxis)],[sigmaprimexiprime(Maxsigma, Maxxis),xidoubleprime(Maxsigma, Maxxis)]])
    VCVmatrix = numpy.matrix([[-sigmadoubleprime(Maxsigma, Maxxis),-sigmaprimexiprime(Maxsigma, Maxxis)],[-sigmaprimexiprime(Maxsigma, Maxxis),-xidoubleprime(Maxsigma, Maxxis)]]).I
    sigmavar = abs(VCVmatrix[0,0])
    xivar = abs(VCVmatrix[1,1])
    sigmaerror = 1.96*math.sqrt(sigmavar)
    xierror = 1.96*math.sqrt(xivar)
    Uppersigma = Maxsigma + sigmaerror
    Lowersigma = Maxsigma - sigmaerror
    Upperxi = Maxxis + xierror
    Lowerxi = Maxxis - xierror
    CIsigma = (Lowersigma, Uppersigma)
    CIxi = (Lowerxi, Upperxi)
if Maxxis == 0:
    Maxxis=.000000001
    numpy.matrix([[sigmadoubleprime(Maxsigma, Maxxis),sigmaprimexiprime(Maxsigma, Maxxis)],[sigmaprimexiprime(Maxsigma, Maxxis),xidoubleprime(Maxsigma, Maxxis)]])
    VCVmatrix = numpy.matrix([[-sigmadoubleprime(Maxsigma, Maxxis),-sigmaprimexiprime(Maxsigma, Maxxis)],[-sigmaprimexiprime(Maxsigma, Maxxis),-xidoubleprime(Maxsigma, Maxxis)]]).I
    sigmavar = abs(VCVmatrix[0,0])
    xivar = abs(VCVmatrix[1,1])
    sigmaerror = 1.96*math.sqrt(sigmavar)
    xierror = 1.96*math.sqrt(xivar)
    Uppersigma = Maxsigma + sigmaerror
    Lowersigma = Maxsigma - sigmaerror
    Upperxi = Maxxis + xierror
    Lowerxi = Maxxis - xierror
    CIsigma = (Lowersigma, Uppersigma)
    CIxi = (Lowerxi, Upperxi)
    Maxxis=0

print '\n'
print 'Parameter Estimates'
print 'u/Threshold = %s' % Threshold
print 'Sigma/Scale = %s' % Maxsigma
print 'Xi/Shape = %s' % Maxxis
print '\n'
print 'Parameter Confidence Intervals'
print 'CI Sigma/Scale = ' + str(CIsigma)
print 'CI Xi/Shape = ' + str(CIxi)
print '\n'

#Distribution
def genparetodist(x, threshold, scale, shape):
    if shape != 0 and 1-(1+shape*(x-threshold)/scale)**(-1/shape)>0:
        return  1-(1+shape*(x-threshold)/scale)**(-1/shape)
    if shape == 0 and 1-math.exp(-((x-threshold)/scale))>0:
        return  1-math.exp(-((x-threshold)/scale))
    if shape != 0 and 1-(1+shape*(x-threshold)/scale)**(-1/shape)<0:
        return  0
    if shape == 0 and 1-math.exp(-((x-threshold)/scale))<0:
        return  0 
        
#quantile... same as genpareto.ppf
def genparetoquantile(percent, threshold, scale, shape):
    if shape !=0:
        return threshold + scale*(percent**(-shape)-1)/shape
    if shape == 0:
        return threshold - scale*math.log(percent)
  
    
#Had to reverse list based on how quantile is defined
listofpercents = [float(i)/(len(yvalues['y'+str(threshposition)])+1) for i in range(1,len(yvalues['y'+str(threshposition)])+1)]        
listofpercentsinverse = [1-float(i)/(len(yvalues['y'+str(threshposition)])+1) for i in range(1,len(yvalues['y'+str(threshposition)])+1)]
#line1 = [x+Threshold for x in range(100000)]
line1 = [Threshold,100000000+Threshold]
quantilelist = [genparetoquantile(percent, Threshold, Maxsigma, Maxxis) for percent in listofpercentsinverse]
quantilelist2 = [genpareto.ppf(percent, Maxxis, loc=Threshold, scale=Maxsigma) for percent in listofpercents]

#User Input Axis for Quantile Plot
print 'Input a plot range for the Quantile Plot'
print 'Click "Display Plot" for various ranges if desired'
print 'Click "Continue" when satisfied with your plot'
print '\n'
quantileaxisrange = []
quantileroot = tk.Tk()
print 'If the graph pops up in a new window you must close out of the graph before displaying a new plot or continuing on with the program.'
print '\n'
def plot_quantile():
    quantileaxisrange.append(int(float(qxmin.get())))
    quantileaxisrange.append(int(float(qxmax.get())))
    quantileaxisrange.append(int(float(qymin.get())))
    quantileaxisrange.append(int(float(qymax.get())))
    #Quantile Plot
    plt.title('Quantile Plot')
    plt.ylabel('Empirical')
    plt.xlabel('Model')
    #Scatter plots are the same... just testing my quantile vs scipy quantile....
    plt.scatter(quantilelist,xvalues['x'+str(threshposition)], s=.5)
    plt.scatter(quantilelist2,xvalues['x'+str(threshposition)], s=.5, label = 'Simulaton Data')
    plt.plot(line1,line1,label='Best Model Fit')
    plt.legend(bbox_to_anchor=(0.,1.1,1.,.102), loc=3,ncol=1,mode="expand",borderaxespad=0.)
    plt.axis([quantileaxisrange[len(quantileaxisrange)-4],quantileaxisrange[len(quantileaxisrange)-3],quantileaxisrange[len(quantileaxisrange)-2],quantileaxisrange[len(quantileaxisrange)-1]])
    plt.show()
    print 'When satisfied with the Quantile Plot, click "Continue".'
    print 'The Dotted line (Real Data), should reasonably agree with the model to make a Linear fit.'
    print '\n'
    
quantileroot.wm_title('Select Axis for the Quantile Plot then click "Display Plot", Once satisfied click "Continue"')
tk.Label(quantileroot, text = "Min x Value").grid(row=0)
tk.Label(quantileroot, text = "Max x Value").grid(row=1)
tk.Label(quantileroot, text = "Min y Value").grid(row=2)
tk.Label(quantileroot, text = "Max y Value").grid(row=3)

qxmin = tk.Entry(quantileroot)
qxmax = tk.Entry(quantileroot)
qymin = tk.Entry(quantileroot)
qymax = tk.Entry(quantileroot)

qxmin.grid(row=0,column=1)
qxmin.insert(10,str(int(0+Threshold)))
qxmax.grid(row=1,column=1)
qxmax.insert(10,str(int(max(xvalues['x'+str(threshposition)])+2)))
qymin.grid(row=2,column=1)
qymin.insert(10,str(int(0+Threshold)))
qymax.grid(row=3,column=1)
qymax.insert(10,str(int(max(xvalues['x'+str(threshposition)])+2)))

tk.Button(quantileroot, text='Display Plot', command=plot_quantile).grid(row=4, column=1, sticky=tk.W, pady=4)
tk.Button(quantileroot, text='Continue', command=quantileroot.destroy).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)

quantileroot.mainloop()

#Probability Plot
line2 = [.01*x for x in range(100)]
plt.title('Probability Plot')
plt.ylabel('Model')
plt.xlabel('Empirical')
plt.scatter(listofpercents,[genparetodist(x, Threshold, Maxsigma, Maxxis) for x in xvalues['x'+str(threshposition)]],s=.5)
plt.scatter(listofpercents,[genpareto.cdf(y, Maxxis, loc=0, scale=Maxsigma) for y in yvalues['y'+str(threshposition)]],s=.5, label='Simulation Data')
plt.axis([0,1,0,1])
plt.plot(line2,line2,'b',label='Best Model Fit')
plt.legend(bbox_to_anchor=(0.,1.1,1.,.102), loc=3,ncol=1,mode="expand",borderaxespad=0.)
plt.show()
print 'The Dotted line (Real Data), should reasonably agree with the model to make a Linear fit.'
print '\n'

    
#Density/Histogram Plot
plt.title('Density Plot')
plt.ylabel('Density')
plt.xlabel('Threshold Exceedances')
plt.hist(numpy.array(xvalues['x'+str(threshposition)]), normed=True, bins=25, facecolor='y',label = 'Simulation Data')
plt.plot(xvalues['x'+str(threshposition)],[genpareto.pdf(xx, Maxxis, loc=Threshold, scale=Maxsigma) for xx in xvalues['x'+str(threshposition)]], 'r-', lw=3,label='Model Fit')
plt.legend(bbox_to_anchor=(0.,1.1,1.,.102), loc=3,ncol=1,mode="expand",borderaxespad=0.)
plt.show()


#Return Level Plot
#Generate the real rplist for the raw data
randproblist = []
rplist = []
for i in range((len(testdata)-len(xvalues['x'+str(threshposition)])),len(testdata)):
    randproblist.append(i/len(testdata))
for probs in randproblist:
    rplist.append((1-probs)**-1)

SPPEU = float(len(yvalues['y'+str(q)]))/float(len(testdata))
threshprobability = float(len(yvalues['y'+str(q)]))/float(len(testdata))

#Return Level Confidence Interval
if Maxxis!=0:
    VCVmatrix = numpy.matrix([[-sigmadoubleprime(Maxsigma, Maxxis),-sigmaprimexiprime(Maxsigma, Maxxis)],[-sigmaprimexiprime(Maxsigma, Maxxis),-xidoubleprime(Maxsigma, Maxxis)]]).I
    VV=numpy.matrix([[threshprobability*(1-threshprobability)/len(testdata),0,0],[0,VCVmatrix[0,0],VCVmatrix[0,1]],[0,VCVmatrix[1,0],VCVmatrix[1,1]]])
    
    #Confidence Interval Delta Method
    def gradxm(m):
        return numpy.matrix([[Maxsigma*(m**Maxxis)*(SPPEU**Maxxis-1),(Maxxis**-1)*((m*SPPEU)**Maxxis-1),-Maxsigma*(Maxxis**-2)*((m*SPPEU)**Maxxis-1)+Maxsigma*(Maxxis**-1)*((m*SPPEU)**Maxxis)*(math.log(m*SPPEU))]])
    #Return Level Plot
    #this effects Range on Return Level Plot, maximum range will be 100000 as listed below. Increase this number to increase the range on the return level plot
    newpercentlist = [float(i)/(100000+1) for i in range(1,100000+1)] #this effects Range on Return Level Plot
    RPs = [(1-p)**-1 for p in newpercentlist]
    #Return Level Function
    def mobsreturnlevel(m, threshold, threshprob, scale, shape):
        if shape!=0:
            return threshold + (scale/shape)*(((m*threshprob)**shape)-1)
        if shape==0:
            return threshold + scale*math.log(m*threshprob)
    mobsrllist = []
    xmci = []
    mobsrllist = [mobsreturnlevel(ms, Threshold, threshprobability, Maxsigma, Maxxis) for ms in RPs]
    xmci = [1.96*math.sqrt(gradxm(mm)*VV*gradxm(mm).T) for mm in RPs]
    upperxm = []
    for i in range(len(mobsrllist)):
        upperxm.append(mobsrllist[i]+xmci[i])
    lowerxm = []
    for i in range(len(mobsrllist)):
        lowerxm.append(mobsrllist[i]-xmci[i])

#IF statement because CI's are different if xis==0
if Maxxis==0:
    #Return Level Plot
    #this effects Range on Return Level Plot, maximum range will be 100000 as listed below. Increase this number to increase the range on the return level plot
    newpercentlist = [float(i)/(100000+1) for i in range(1,100000+1)] #this effects Range on Return Level Plot
    RPs = [(1-p)**-1 for p in newpercentlist]
    def mobsreturnlevel(m, threshold, threshprob, scale, shape):
        if shape!=0:
            return threshold + (scale/shape)*(((m*threshprob)**shape)-1)
        if shape==0:
            return threshold + scale*math.log(m*threshprob)
    mobsrllist = []
    xmci = []
    mobsrllist = [mobsreturnlevel(ms, Threshold, threshprobability, Maxsigma, Maxxis) for ms in RPs]
    upperxm = []
    lowerxm = []
    for xm,m in zip(mobsrllist,RPs):
        xmvar = ((xm-Threshold)/(math.log(m*threshprobability)))**2
        xmerror = 1.96*math.sqrt(abs(xmvar))
        upperxm.append(xm + xmerror)
        lowerxm.append(xm - xmerror)
        

#Return Level Plot
print 'Input a plot range for the Return Level Plot'
print 'Click "Display Plot" for various ranges if desired'
print 'Click "Continue" when satisfied with your plot'
print '\n'
rlaxisrange = []
rlroot = tk.Tk()
print 'If the graph pops up in a new window you must close out of the graph before displaying a new plot or continuing on with the program.'

def plot_rl():
    rlaxisrange.append(int(float(rlqxmin.get())))
    rlaxisrange.append(int(float(rlqxmax.get())))
    rlaxisrange.append(int(float(rlqymin.get())))
    rlaxisrange.append(int(float(rlqymax.get())))
    plt.title('Return Level Plot')
    plt.ylabel('Return Level')
    plt.xlabel('Return Period')
    plt.plot(RPs,mobsrllist,label='Best Model Fit')
    plt.plot(RPs,lowerxm,'r--')
    plt.plot(RPs,upperxm,'r--',label='Upper and Lower Confidence Intervals')
    plt.xlim([rlaxisrange[len(rlaxisrange)-4],rlaxisrange[len(rlaxisrange)-3]])
    plt.ylim([rlaxisrange[len(rlaxisrange)-2],rlaxisrange[len(rlaxisrange)-1]])
    plt.scatter(rplist,xvalues['x'+str(threshposition)],s=2,label='Simulation Data')
    plt.legend(bbox_to_anchor=(0.,1.1,1.,.102), loc=3,ncol=1,mode="expand",borderaxespad=0.)
    plt.xscale('log')
    plt.show()
    print 'When satisfied with the Return Level Plot, click "Continue".'
    print 'The Dotted line (Real Data), should reasonably agree with the model and follow the blue curve. The Real Data should also fall within the red dashed lines (95% Confidence Intervals).'


rlroot.wm_title('Select Axis for the Return Level Plot then click "Display Plot", Once satisfied click "Continue"')
tk.Label(rlroot, text = "Min x Value").grid(row=0)
tk.Label(rlroot, text = "Max x Value").grid(row=1)
tk.Label(rlroot, text = "Min y Value").grid(row=2)
tk.Label(rlroot, text = "Max y Value").grid(row=3)

rlqxmin = tk.Entry(rlroot)
rlqxmax = tk.Entry(rlroot)
rlqymin = tk.Entry(rlroot)
rlqymax = tk.Entry(rlroot)

rlqxmin.grid(row=0,column=1)
rlqxmin.insert(10,str(int(min(rplist))))
rlqxmax.grid(row=1,column=1)
rlqxmax.insert(10,str(int(max(rplist))))
rlqymin.grid(row=2,column=1)
rlqymin.insert(10,str(int(min(xvalues['x'+str(threshposition)]))))
rlqymax.grid(row=3,column=1)
rlqymax.insert(10,str(int(max(xvalues['x'+str(threshposition)]))))

tk.Button(rlroot, text='Display Plot', command=plot_rl).grid(row=4, column=1, sticky=tk.W, pady=4)
tk.Button(rlroot, text='Continue', command=rlroot.destroy).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)

rlroot.mainloop()


#When Calculating Return Periods from Probabilities:
#Multiply the calculated RP by the minimum RP for the system...
#Then that pair of probability and RP will give corresponding Quantile and RL
#User Input to dertermine Force Value based on a probability and Return Level based on Return Period
print '\n'
print 'Input a Probability or a Return Period to calculate a Force or Return Level.'
print '\n'
print 'The "Calculate Force" button predicts with a certain probability that the force in the flexor will be less than or equal to the output force value (based on the input probability).'
print '\n'
print 'The "Calculate Return Level and Confidence Interval" button takes a Return Period (number of observations of data) and returns the Return Level that is exceeded on average based on the Return Period Observations.'
print '\n'
problist = []
mcilist = []
root4 = tk.Tk()
def getprob():
    print("Probability: %s" % entryy1.get())
    print("a Probability of: %s corresponds with a Return Period of %s") % ((entryy1.get()),min(rplist)*(1-float(entryy1.get()))**-1)
    problist.append(float(entryy1.get()))
    mentry.delete(0,'end')
    mentry.insert(10,str(min(rplist)*(1-float(entryy1.get()))**-1))
    print 'Force = ' + str(genparetoquantile(1-problist[len(problist)-1], Threshold, Maxsigma, Maxxis))
    print 'Force = ' + str(genpareto.ppf(problist[len(problist)-1], Maxxis, loc=Threshold, scale=Maxsigma))
    print '\n'
if Maxxis!=0:
    def getCI():
        print("Return Period = %s" % mentry.get())
        mcilist.append(float(mentry.get()))
        userxm = mobsreturnlevel(mcilist[-1], Threshold, threshprobability, Maxsigma, Maxxis)
        userxmci = 1.96*math.sqrt(gradxm(mcilist[-1])*VV*gradxm(mcilist[len(mcilist)-1]).T)
        userupperxm = userxm+userxmci
        userlowerxm = userxm-userxmci
        ciforprint = (userlowerxm,userupperxm)
        print 'Return Level = ' + str(userxm)
        print '95% Confidence Interval = ' + str(ciforprint)
        print '\n'
if Maxxis==0:
    def getCI():
        print("Return Period = %s" % mentry.get())
        mcilist.append(float(mentry.get()))
        userxm = mobsreturnlevel(mcilist[-1], Threshold, threshprobability, Maxsigma, Maxxis)
        userxmci = 1.96*math.sqrt(((xm-Threshold)/(math.log(m*threshprobability)))**2)
        userupperxm = userxm+userxmci
        userlowerxm = userxm-userxmci
        ciforprint = (userlowerxm,userupperxm)
        print 'Return Level = ' + str(userxm)
        print '95% Confidence Interval = ' + str(ciforprint)
        print '\n'
root4.wm_title('Input a Probability/Return Level, then click "Calculate Force/Calculate Return Level and Confidence Interval"')
tk.Label(root4, text = "Input a Return Period").grid(row=2)
mentry = tk.Entry(root4)
mentry.grid(row=2,column=1)
tk.Button(root4, text='Calculate Return Level and Confidence Interval', command=getCI).grid(row=2, column=2, sticky=tk.W, pady=4)
tk.Label(root4, text = "Input a Probability").grid(row=1)
entryy1 = tk.Entry(root4)
entryy1.grid(row=1,column=1)
entryy1.insert(10,".95")
mentry.insert(10,str(min(rplist)*(1-float(entryy1.get()))**-1))
tk.Button(root4, text='Calculate Force', command=getprob).grid(row=1, column=2, sticky=tk.W, pady=4)
tk.Button(root4, text='Continue', command=root4.destroy).grid(row=3, column=0, sticky=tk.W, pady=4, padx=4)
root4.mainloop()


#Return Period = 1e7
#Return Level = 5085973.42376
#95% Confidence Interval = (2413063.8381898752, 7758883.009328419)

#Return Period = 1e6
#Return Level = 4280948.97564
#95% Confidence Interval = (2643728.3434504466, 5918169.607830725)



################################################################################################
################################################################################################
################################################################################################
#Testing LLF with Return Levels to generate more accurate confidence intervals
################################################################################################
################################################################################################
################################################################################################
#
#print str('   xis') + '               '  + str('x') + '             ' +str('Approx Zero')+ '         ' +str('LLF Value')
#delta = threshprobability
#u = Threshold
#Define a specific m (Return Period)
#m = 100
#Calculate the corresponding system xm (Return Level)
#sxm = mobsreturnlevel(m, Threshold, threshprobability, Maxsigma, Maxxis)
#Generate a list of x's in an interval around sxm
#xmtest = [sxm+5*i for i in range(-20,20)]
#xislist = []
#LLxmlist = []
#For each xm/x in the list of x's we plug x/xm into the derivative of the LLF WRT xis(shape) and maximize
#for xm in xmtest:
#    def LLDxmxis(xi):
#        if xi !=0:
#            return -len(yvalues['y'+str(q)])*((delta*m)**xi - 1)*(-xi*(delta*m)**xi*(-u + xm)*math.log(delta*m)/((delta*m)**xi - 1)**2 + (-u + xm)/((delta*m)**xi - 1))/(xi*(-u + xm)) - (1 + 1/xi)*sum(yvalues['y'+str(q)][w]*(delta*m)**xi*math.log(delta*m)/((-u + xm)*(yvalues['y'+str(q)][w]*((delta*m)**xi - 1)/(-u + xm) + 1)) for w in range(len(yvalues['y'+str(q)]))) + sum(math.log(yvalues['y'+str(q)][w]*((delta*m)**xi - 1)/(-u + xm) + 1) for w in range(len(yvalues['y'+str(q)])))/xi**2
#        if xi == 0:
#            return -len(yvalues['y'+str(q)])*((delta*m)**xi - 1)*(-xi*(delta*m)**xi*(-u + xm)*math.log(delta*m)/((delta*m)**xi - 1)**2 + (-u + xm)/((delta*m)**xi - 1))/(xi*(-u + xm)) + (delta*m)**xi*math.log(delta*m)*sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/(xi*(-u + xm)*((delta*m)**xi - 1)**2) + sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/(xi**2*(-u + xm)*((delta*m)**xi - 1))
#    def LLxm(xis,xm,m):
#        if xis != 0 and xis != -1:
#            return -len(yvalues['y'+str(q)])*math.log(((xm-u)*xis)/(((m*delta)**xis)-1)) - (1 + 1/xis)*sum(math.log(yvalues['y'+str(q)][w]*xis/((xm-u)*xis)/(((m*delta)**xis)-1) + 1) for w in range(len(yvalues['y'+str(q)])))
#       if xis == 0:
#            return -len(yvalues['y'+str(q)])*math.log(((xm-u)*xis)/(((m*delta)**xis)-1)) - sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/((xm-u)*xis)/(((m*delta)**xis)-1)
#    #This is the optimization here... for each x/xm in xmtest we find the xis(shape) that makes the derivative of LLF zero or maximizes the LLF
#    solvedxis = fsolve(LLDxmxis,-.001)
#    #Create list of all shapes
#    xislist.append(float(solvedxis))
#    LLxmvalue = LLxm(solvedxis,xm,m)
#    #Create a list of all Log-Likelihood Values
#    LLxmlist.append(LLxmvalue)
#    print str(float(solvedxis)) + '   '  + str(xm) + '   ' +str(LLDxmxis(solvedxis))+ '   ' +str(LLxmvalue)

#print str('   xis') + '               '  + str('x') + '             ' +str('Approx Zero')+ '         ' +str('LLF Value')

#Find Maximum of the LLF and associated xis/shape and x/xm
#MaxLLxm = max(LLxmlist)
#MaxLLxmindex = LLxmlist.index(max(LLxmlist))
#Maxxisxm = xislist[MaxLLxmindex]
#Maxxm = xmtest[MaxLLxmindex]

#plot range of x's against LLF
#Return Level vs Log-Likelihood Plot
#plt.title('Shape vs Log-Likelihood')
#plt.ylabel('Profile Log-Likelihood')
#plt.xlabel('Return Level')
#plt.plot(xmtest,LLxmlist)
#plt.show()

#Trying to find root
#xilist = []
#xmlist = []
#xilist = [.1*i for i in range(1,110)]
#xmlist = [LLDxmxis(i) for i in xilist]
#plt.xlim([0,100])
#plt.plot(xilist,xmlist)

