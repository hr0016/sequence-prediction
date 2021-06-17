import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import warnings
import matplotlib.pyplot as plt

class KneeFinder():
    '''
    Write docstring - Find out what it should contain for the class
    '''
    
    def __init__(self, cycles, y, truncate=False, mode='knee'):
        self.cycles = cycles
        self.y = y
        self.truncate = truncate # whether or not to truncate using sigmoid fit
        self.mode = mode # knee or elbow
        self.point = None
        self.onset = None
        self.eol_reached = False
        self.eol_cycle = None
        
        # Set the fitting parameters to their default values
        self.reset_all_parameters()
        # Get a continuous array of integer cycle numbers
        self.x_cont = self.process_cycles_array(self.cycles)
    
    
    # Methods
    # Set up the models - should these 4 be staticmethods?
    def _asym_sigmoidal(self, x, a, b, c, d, m):
        ''' Formula for asymmetric sigmoidal function '''
        # Ignore the warnings about overflow in power
        warnings.filterwarnings(
            action='ignore',
            message='overflow encountered in power',
            module=r'.*knee_finder')
        return d + ((a - d) / (1 + (x / c) ** b) ** m)
    
    
    def _bw_func(self, x, b0, b1, b2, cp):
        ''' Formula for the single Bacon-Watts model'''
        return b0 + b1*(x-cp) + b2*(x-cp)*np.tanh((x-cp)/1e-8)
    
    
    def _bw2_func(self, x, b0, b1, b2, b3, cp, co):
        ''' Formula for the double Bacon-Watts model'''
        return b0 + b1*(x-cp) + b2*(x-cp)*np.tanh((x-cp)/1e-8) + b3*(x-co)*np.tanh((x-co)/1e-8)

    
    def _exponential(self, x, a, b, c, d, theta):
        ''' Formula for the line plus exponential model'''
        # Ignore the warnings about overflow in power
        warnings.filterwarnings(
            action='ignore',
            message='overflow encountered in multiply',
            module=r'.*knee_finder')
        warnings.filterwarnings(
            action='ignore',
            message='overflow encountered in exp',
            module=r'.*knee_finder')
        return d*np.exp(a*x - b) + c + theta*x
    
    
    def find_onset_and_point(self):
        '''
        Write docstring
        '''
        # Get the monotonic fit to the experimental data
        self.mon_data = self.fit_monotonic(x_cont=self.x_cont,
                                           x_data=self.cycles,
                                           y_data=self.y)
        
        # Fit asym_sigmoid to the monotonic data if self.truncate==True       
        if self.truncate:
            # Try to fit the sigmoid, but this may fail with a RuntimeError
            # telling us that the optimal parameters were not found.
            try:
                self.sig_fit = self.fit_sigmoid(x_data=self.x_cont,
                                                 y_data=self.mon_data)
                
                self.indices = self.get_truncated_indices(data=self.sig_fit)
                
            except RuntimeError as err:
                error_msg = str(err)
                print(error_msg)
                # Check for the specific error message about fitting
                if 'Optimal parameters not found' in error_msg:
                    print("Can't fit sigmoid for truncation. Skipping")
                # Since we can't fit the asym_sigmoid for truncation,
                # set self.indices as if self.truncate==False
                self.indices = np.arange(len(self.mon_data))
        else:
            self.indices = np.arange(len(self.mon_data))
        
        # Fit line_exponential to the (normal/truncated) monotonic
        self.exp_fit = self.fit_line_exp(x_data=self.x_cont,
                                         y_data=self.mon_data,
                                         indices=self.indices)
        
        # Fit BW and double BW to exp_fit
        self.point, self.bw_fit = self.fit_bacon_watts(x_cont=self.x_cont,
                                                       indices=self.indices,
                                                       y_data=self.exp_fit)
        
        self.onset, self.bw2_fit = self.fit_double_bacon_watts(x_cont=self.x_cont,
                                                               indices=self.indices,
                                                               y_data=self.exp_fit)
        
        
        # Find the indices of the values in x_cont that are closest to the
        # cycle numbers for onset and point, as computed using BW and double BW
        onset_idx = np.argmin(np.abs(self.x_cont - int(self.onset)))
        point_idx = np.argmin(np.abs(self.x_cont - int(self.point)))
        # Get the y value (capacity/IR) corresponding to onset and point,
        # using mon_data
        self.onset_y = self.mon_data[onset_idx]
        self.point_y = self.mon_data[point_idx]
        
        # Include a return statement so you can use line-profiler to assess running time
        return
    
    
    def find_eol(self, nominal=None):
        '''
        Determine whether or not end of life (EOL) is reached,
        and if so, at which cycle.
        
        Use the monotonic fit to check for EOL. This is because:
            1. The values are within the range of experimental values. This
                is not the case if you were to extend the line_exp fit past 
                the truncation point. It decreases very rapidly, so you are
                almost sure to find an EOL, even though it doesn't occur in
                the actual experimental data.
            2. Putting (1) aside, if you find an EOL using the truncated or
                un-truncated line_exp fit, there is often a mismatch between
                the cycle number at which line_exp reaches 80% and the cycle
                number at which the monotonic fit reaches 80%. The monotonic
                fit, being essentially linear interpolation between points,
                much more closely fits the experimental data, so the EOL cycle
                number identified using monotonic fit is more reliable.
            3. If you use the truncated line_exp fit, you are potentially
                ignoring a large percentage of the data (past the truncation).
                You could then easily miss EOL occurrences from past the cycle
                at which the line_exp fit is truncated. Note, simply extending
                the line_exp fit past the truncation point is covered in (1).
        '''
        # Begin by assuming EOL is not reached
        self.eol_reached = False
        
        # Get the monotonic fit
        self.mon_data = self.fit_monotonic(x_cont=self.x_cont,
                                           x_data=self.cycles,
                                           y_data=self.y)
        
        # Use the monotonic fit to check for EOL, because it is guaranteed
        # to be there for the whole curve and it's essentially linear interp
        # meaning that it will be closer to the data than line_exp or sig fits
        
        ## New code. If a value is specified for nominal, use this instead
        ## of initial experimental capacity value for checking for EOL.
        if nominal != None:
            init_val = nominal
        else:
            # Find the initial (experimental) value and compute the EOL value
            init_val = self.y[0]
            
            
        if self.mode.lower() == 'knee':
            self.eol_val = 0.8 * init_val
            self.post_eol_indices = np.where(self.mon_data < self.eol_val)[0]
        
        elif self.mode.lower() == 'elbow':
            self.eol_val = 2.0 * init_val
            self.post_eol_indices = np.where(self.mon_data > self.eol_val)[0]
    
        # If it finds indices past the EOL index, set eol_reached to true and
        # find the first cycle number in x_cont after the EOL value is reached
        if len(self.post_eol_indices) > 0:
            self.eol_reached = True
            self.eol_idx = self.post_eol_indices[0]
            self.eol_cycle = self.x_cont[self.eol_idx]
    
        return
    
    
    # Define a method to show the results on a plot
    def plot_results(self, line_exp=False, mon=False, data_style='-'):
        '''
        Write docstring
        '''
        fig, ax = plt.subplots()
        ax.plot(self.cycles, self.y, data_style, label='Experimental')
        if mon:
            ax.plot(self.x_cont, self.mon_data, label='Monotonic', color='green')
        if line_exp:
            ax.plot(self.x_cont[self.indices], self.exp_fit, label='Line_exp fit', color='purple')
        ax.axvline(self.onset, label=f'{self.mode.capitalize()} onset', color='orange')
        ax.axvline(self.point, label=f'{self.mode.capitalize()} point', color='red')
        ax.axhline(self.onset_y, color='orange')
        ax.axhline(self.point_y, color='red')
        if self.eol_cycle != None:
            ax.axvline(self.eol_cycle, label='End of life', color='black')
        ax.grid(alpha=0.4)
        ax.legend()
        plt.show()
        
    
    # Define the helper methods
    def process_cycles_array(self, cycles):
        '''
        Write docstring        
        '''
        
        # Identify the first and last cycle numbers in the cycles array.
        # In the case of non-integer cycle values, round up to the next
        # integer for the first cycle and remove any fractional part of
        # the last cycle number. Avoids issues with NaNs in the monotonic fit.
        first_cycle = int(np.ceil(cycles[0]))
        last_cycle = int(cycles[-1])
        
        # Create an array of integers from first to last cycle numbers.
        x_cont = np.arange(first_cycle, last_cycle+1)
        
        return x_cont
    
    
    def fit_monotonic(self, x_cont, x_data, y_data):
        '''
        x_cont (type: array)
            Array of continuous integer values for which the fitted
            IsotonicRegression result should be used to generate the monotonic
            fit.
            
        x_data (type: array)
            Experimental x data e.g. cycle number
        
        y_data (type: array)
            Experimental y data e.g. capacity or internal resistance
        
        '''
        # Create an IsotonicRegression instance
        if self.mode.lower() == 'knee':
            ir = IsotonicRegression(increasing=False)
        elif self.mode.lower() == 'elbow':
            ir = IsotonicRegression(increasing=True)

        # Fit the monotonic curve to the experimental data
        ir.fit(x_data, y_data)
        
        # Get a value for every cycle based on the fit
        mon_data = ir.predict(x_cont)
        
        return mon_data
    
    
    def fit_sigmoid(self, x_data, y_data):
        '''
        Write docstring
        '''
        sig_popt, _ = curve_fit(self._asym_sigmoidal,
                               x_data,
                               y_data,
                               p0=self.sig_p0,
                               bounds=self.sig_bounds)
        
        sig_fit = self._asym_sigmoidal(x_data, *sig_popt)

        return sig_fit
    
    
    def compute_second_derivative(self, data):
        '''
        Write docstring
        '''
        
        # Using np.gradient gives us a result of the same shape
        dy_dx = np.gradient(data)
        d2y_dx2 = np.gradient(dy_dx)
        
        # Set a threshold, below which, the value is set to zero
        d2y_dx2[np.where(np.abs(d2y_dx2) < 1e-10)] = 0.0
        
        # Replace the last 2 values with the value at index [-3] to avoid
        # the sharp change that happens at the end of the d2y_dx2 array
        d2y_dx2[-2] = d2y_dx2[-3]
        d2y_dx2[-1] = d2y_dx2[-3]
        
        return d2y_dx2
       
    
    def get_truncated_indices(self, data):
        '''
        Write docstring
        '''
        # Compute the second derivative of data,
        # which will be the asymmetric sigmoid fit
        self.d2 = self.compute_second_derivative(data)
        # Filter d2
        self.d2 = medfilt(self.d2, 5)
        # Get an array of the sign of d2 to find changes
        self.d2_sign = np.sign(self.d2)
        
        # Use mode to determine which new sign value to look for to find changes
        if self.mode == 'knee':
            new_sign_val = 1
        elif self.mode == 'elbow':
            new_sign_val = -1
            
        # Find out if there are any places where the sign changes
        change_indices = np.where(self.d2_sign == new_sign_val)[0]
        # If there are no sign changes, return the full array of indices
        if len(change_indices) == 0:
            indices = np.arange(0, len(data))
            return indices
        else:
            # Find the first index at which the sign changes from -ve to +ve
            self.first_change_idx = change_indices[0]
            
            # If first_change_idx is in the last 20 indices, don't look ahead
            if len(data) - self.first_change_idx > 20:
                # Look ahead a few cycles to make sure it is not a local erroneous fluctuation
                # Note this is incomplete, because there's nothing to handle AssertionErrors
                lookahead = np.min((20, len(self.d2_sign) - self.first_change_idx))
                assert(np.all(self.d2_sign[self.first_change_idx + lookahead] == new_sign_val))
            indices = np.arange(0, self.first_change_idx)

        return indices
    
    
    def fit_line_exp(self, x_data, y_data, indices):
        '''
        Write docstring
        '''
        # Use the indices array to select subsets of the other input arrays
        # if truncation has been deemed necessary. If no truncation is needed,
        # the indices array will contain the indices for every cycle
        x_data = x_data[indices]
        y_data = y_data[indices]
        
        # Fit the line plus exponential model to the monotonic data
        # Get the optimal parameters for _exponential
        exp_popt, _ = curve_fit(self._exponential,
                                x_data,
                                y_data,
                                p0=self.line_exp_p0,
                                bounds=self.line_exp_bounds)
        # Apply the optimal parameters to the continuous cycle array
        exp_fit = self._exponential(x_data, *exp_popt)
        
        return exp_fit
    
    
    def fit_bacon_watts(self, x_cont, indices, y_data):
        '''
        Write docstring.
        
        Pass the line_exponential fit to this method, to use the
        Bacon-Watts method to compute the cycle number at which
        the knee point occurs.
        '''
        
        # Use the indices array to select subsets of the other input arrays
        x_cont = x_cont[indices]
        y_data = y_data[indices]
        
        # Set some p0 and bounds values based on the cycle numbers
        # of the data being considered by the KneeFinder instance
        self.bw_p0[3] = x_cont[-1]/1.5
        # Set the upper bound to be the final cycle
        self.bw_bounds[1][3] = x_cont[-1]

        # Fit the Bacon Watts model to the line_exponential input
        popt_bw, _ = curve_fit(self._bw_func,
                               x_cont,
                               y_data,
                               p0=self.bw_p0,
                               bounds=self.bw_bounds)
        
        # Apply the optimal parameters to the continuous cycle array
        # to get the Bacon-Watts fit
        bw_fit = self._bw_func(x_cont, *popt_bw)
        
        return popt_bw[3], bw_fit


    def fit_double_bacon_watts(self, x_cont, indices, y_data):
        '''
        Write docstring
        '''
        # Use the indices array to select subsets of the other input arrays
        x_cont = x_cont[indices]
        y_data = y_data[indices]
        
        # Set some p0 and bounds values based on the cycle numbers
        # of the data being considered by the KneeFinder instance.
        # These p0 values give the onset and point a starting location
        # somewhere in the second half of the curve.
        self.bw2_p0[4] = x_cont[-1]/2.0
        self.bw2_p0[5] = x_cont[-1]/1.5
        # Set the upper bound to be the final cycle
        self.bw2_bounds[1][4] = x_cont[-1]
        self.bw2_bounds[1][5] = x_cont[-1]
        

        # Apply the optimal parameters to the continuous cycle array
        # to get the double Bacon-Watts fit
        popt_bw2, _ = curve_fit(self._bw2_func,
                               x_cont,
                               y_data,
                               p0=self.bw2_p0,
                               bounds=self.bw2_bounds)
        
        bw2_fit = self._bw2_func(x_cont, *popt_bw2)
        
        return popt_bw2[4], bw2_fit
        
    
    # Methods for setting and resetting parameters. All of these are very similar
    def reset_all_parameters(self):
        try:
            # Set the parameters to their default values
            self.line_exp_p0 = [1, 1, 1, 1, 1]
            self.line_exp_bounds = [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]
            self.sig_p0 = [1, 1, 1, 1, 1]
            self.sig_bounds = [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]
            self.bw_p0 = [1, 1, 1, 1]
            self.bw_bounds = ([-np.inf, -np.inf, -np.inf, 0],[np.inf, np.inf, np.inf, np.inf])
            self.bw2_p0 = [1, 1, 1, 1, 1, 1]
            self.bw2_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        
        except Exception as e:
            print(f"Error: {e}")
    
    
    def set_params_using_dict(self, params_dict, src, data_type):
        try:
            # Assign the parameter values for the fitting functions,
            # for this particular instance.
            self.set_sigmoid_params(params_dict[src][data_type]['sig']['p0'])
            self.set_sigmoid_bounds(params_dict[src][data_type]['sig']['bounds'])
            self.set_line_exp_params(params_dict[src][data_type]['line_exp']['p0'])
            self.set_line_exp_bounds(params_dict[src][data_type]['line_exp']['bounds'])
        
        except Exception as e:
            print(f"Error: {e}")
            
    
    def set_line_exp_params(self, p0=None):
        # Input validation - check for the correct lengths.
        # Could also check that they obey the relevant bounds,
        # but could maybe leave that to common sense or leave the
        # curve_fit method to raise the exception.
        try:
            assert(len(p0)==5)
            self.line_exp_p0 = p0
            
        except AssertionError:
            print("Argument p0 must have length of 5 for line exponential.")
            print(p0)
            return None
        
        except Exception as e:
            print(f"Error: {e}")
        
        
    def set_line_exp_bounds(self, bounds=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(bounds)==2)
            assert(len(bounds[0])==5)
            assert(len(bounds[1])==5)
            self.line_exp_bounds = bounds
            
        except AssertionError:
            print("bounds must have length of 5 for line exponential.")
            print(bounds)
            return None
        
        except Exception as e:
            print(f"Error: {e}")
            
    
    def reset_line_exp_params(self):
        try:
            self.line_exp_p0 = [1, 1, 1, 1, 1]
            self.line_exp_bounds = [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]
            print("Parameters for line_exp reset")
        
        except Exception as e:
            print(f"Error: {e}")
        
        
    def set_sigmoid_params(self, p0=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(p0)==5)
            self.sig_p0 = p0
            
        except AssertionError:
            print("Argument p0 must have length of 5 for asym_sigmoid.")
            print(p0)
            return None
        
        except Exception as e:
            print(f"Error: {e}")
        
            
    def set_sigmoid_bounds(self, bounds=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(bounds)==2)
            assert(len(bounds[0])==5)
            assert(len(bounds[1])==5)
            self.sig_bounds = bounds
            
        except AssertionError:
            print("bounds must have length of 5 for asym_sigmoid.")
            print(bounds)
            return None
        
        except Exception as e:
            print(f"Error: {e}")
    
    
    def reset_sigmoid_params(self):
        try:
            self.sig_p0 = [1, 1, 1, 1, 1]
            self.sig_bounds = [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]
            print("Parameters for asym_sigmoid reset")
        
        except Exception as e:
            print(e)
        
        
    def set_bw_params(self, p0=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(p0)==4)
            self.bw_p0 = p0
            
        except AssertionError:
            print("Argument p0 must have length of 4 for Bacon Watts.")
            print(p0)
            
        except Exception as e:
            print(f"Error: {e}")
            
            
    def set_bw_bounds(self, bounds=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(bounds)==2)
            assert(len(bounds[0])==4)
            assert(len(bounds[1])==4)
            self.bw_bounds = bounds
            
        except AssertionError:
            print("bounds must have length of 4 for Bacon Watts.")
            print(bounds)
            return None
        
        except Exception as e:
            print(f"Error: {e}")
    
    
    def reset_bw_params(self):
        try:
            # Reset the Bacon-Watts initial parameters and bounds to
            # some default values
            self.bw_p0 = [1, 1, 1, 1]
            self.bw_bounds = ([-np.inf, -np.inf, -np.inf, 0],
                              [np.inf, np.inf, np.inf, np.inf])
            print("Parameters for Bacon Watts reset")
        
        except Exception as e:
            print(f"Error: {e}")
            
        
    def set_bw2_params(self, p0=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(p0)==6)
            self.bw2_p0 = p0
            
        except AssertionError:
            print("Arguments p0 and bounds must have length of 6 for double Bacon Watts.")
            print(p0)
            
        except Exception as e:
            print(f"Error: {e}")
    
    
    def set_bw2_bounds(self, bounds=None):
        # Input validation - check for the correct lengths
        try:
            assert(len(bounds)==2)
            assert(len(bounds[0])==6)
            assert(len(bounds[1])==6)
            self.bw2_bounds = bounds
        
        except AssertionError:
            print("bounds must have length of 6 for double Bacon Watts.")
            print(bounds)
            
        except Exception as e:
            print(f"Error: {e}")
        
    
    def reset_bw2_params(self):
        try:
            # Reset the double Bacon-Watts initial parameters and bounds to
            # some default values
            self.bw2_p0 = [1, 1, 1, 1, 1, 1]
            self.bw2_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0],
                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
            print("Parameters for double Bacon Watts reset")
            
        except Exception as e:
            print(f"Error: {e}")