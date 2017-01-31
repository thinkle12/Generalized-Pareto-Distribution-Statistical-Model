# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:49:08 2016

@author: Trent
"""

from distutils.core import setup
import py2exe
import matplotlib

excludes = []
includes = ["scipy.special._ufuncs_cxx", "scipy.sparse.csgraph._validation","matplotlib.backends.backend_qt4agg"]
opts = {"py2exe": {"includes":includes,"excludes":excludes,"dll_excludes":["MSVCP90.dll"]}}
setup(console=['ParetoModel.py'],
      options=opts,
      data_files=matplotlib.get_py2exe_datafiles())
      
#    if xis<-1:
#        def LLDsigma(sigma):
#            if xis != 0:
#                return -len(yvalues['y'+str(q)])/sigma - (1 + 1/xis)*sum(-yvalues['y'+str(q)][w]*xis/(sigma**2*(yvalues['y'+str(q)][w]*xis/sigma + 1)) for w in range(len(yvalues['y'+str(q)])))
#            if xis == 0:
#                return -len(yvalues['y'+str(q)])/sigma + sum(yvalues['y'+str(q)][w] for w in range(len(yvalues['y'+str(q)])))/sigma**2
#        potentialsigmalist = []
#        minimizerlist = []
#        try:
#            potentialsigmalist.append(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)))
#        except ValueError:
#            for i in range(1,100):
#                try:
#                    potentialsigmalist.append(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)-.01*i))
#                except ValueError:
#                    'Value Error'
#                except ZeroDivisionError:
#                    'Zero Division Error'
#        if len(potentialsigmalist)==1:
#            for i in range(1,100):
#                    try:
#                        potentialsigmalist.append(scipy.optimize.bisect(LLDsigma,abs(max(yvalues['y'+str(threshposition)])),abs(max(yvalues['y'+str(threshposition)])*xis)-.01*i))
#                    except ValueError:
#                        'Value Error'
#                    except ZeroDivisionError:
#                        'Zero Division Error'
#        for potential in potentialsigmalist:
#            minimizerlist.append(LLDsigma(potential))
#        solvedsigma = minimizerlist.index(min(minimizerlist))
#        sigmalist.append(float(solvedsigma))  
      
#attempting make dynamic density/histogram plot that can take away outliers
#removeoutlierlist = []
#removethesevalues = []
#histogramroot = tk.Tk()
#def plot_histo():
#    if numberofoutliers.get()>0:
#        listofdeletions = []
#        newxvalues = xvalues['x'+str(threshposition)]
#        for i in range(5):
#            listofdeletions.append(newxvalues[-1-i])    
#        for deletions in listofdeletions:
#            newxvalues.remove(deletions)
#        plt.title('Density Plot')
#        plt.ylabel('Density')
#        plt.xlabel('Threshold Exceedances')
#        plt.hist(numpy.array(newxvalues), normed=True, bins=25, facecolor='y')
#        plt.plot(newxvalues,[genpareto.pdf(xx, Maxxis, loc=Threshold, scale=Maxsigma) for xx in newxvalues], 'r-', lw=3)
#        plt.show()
#    if numberofoutliers.get()==0:
#        newxvalues = []
#        newxvalues = xvalues['x'+str(threshposition)]
#        plt.title('Density Plot')
#        plt.ylabel('Density')
#        plt.xlabel('Threshold Exceedances')
#        plt.hist(numpy.array(newxvalues), normed=True, bins=25, facecolor='y')
#        plt.plot(newxvalues,[genpareto.pdf(xx, Maxxis, loc=Threshold, scale=Maxsigma) for xx in newxvalues], 'r-', lw=3)
#        plt.show()
    
#histogramroot.wm_title('Select the number of outliers to remove, then click "Display Plot", Once satisfied click "Continue"')
#tk.Label(histogramroot, text = "Number of outliers to remove").grid(row=0)
#numberofoutliers = tk.Entry(histogramroot)
#numberofoutliers.grid(row=0,column=1)
#numberofoutliers.insert(10,"0")

#tk.Button(histogramroot, text='Display Plot', command=plot_histo).grid(row=4, column=1, sticky=tk.W, pady=4)
#tk.Button(histogramroot, text='Continue', command=histogramroot.destroy).grid(row=4, column=0, sticky=tk.W, pady=4, padx=4)

#histogramroot.mainloop()