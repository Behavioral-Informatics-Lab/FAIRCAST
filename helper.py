import numpy as np
import pandas as pd


### Just to print the empirical probability
def print_probs(df, verbose = False):
    
    ## conditinoal P(Y | S)
    p_1_1 = (df.query('y == 1 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0])
    p_1_0 = (df.query('y == 1 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0])
    p_0_1 = (df.query('y == 0 and sensitive == 1').shape[0] / df.query('sensitive == 1').shape[0])
    p_0_0 = (df.query('y == 0 and sensitive == 0').shape[0] / df.query('sensitive == 0').shape[0])

    if verbose:
        print(p_1_1)
        print(p_1_0)
        print(p_0_1)
        print(p_0_0)

        print('\n')
        ## joint
        print(df.query('y == 1 and sensitive == 1').shape[0] / df.shape[0])
        print(df.query('y == 1 and sensitive == 0').shape[0] / df.shape[0])
        print(df.query('y == 0 and sensitive == 1').shape[0] / df.shape[0])
        print(df.query('y == 0 and sensitive == 0').shape[0] / df.shape[0])

        print('\n')
        
        print(df.query('sensitive == 1').shape[0] / df.shape[0])
        print(df.query('sensitive == 0').shape[0] / df.shape[0])
        
        print('\n')
        
        print(df.query('y == 1').shape[0] / df.shape[0])
        print(df.query('y == 0').shape[0] / df.shape[0])
        
 

    return np.array([p_1_1, p_1_0, p_0_1, p_0_0])

def generate_sample_label_shift(data, rates, n_samples):
    '''
        - Simulate a label shift based on the y values
        - Just P(Y) changes 
        - Number of samples are fixed
    
    '''
    assert np.sum(rates) == 1.0
    S0 = data[data['y'] == 0].sample(int(n_samples * rates[0]))
    S1 = data[data['y'] == 1].sample(int(n_samples * rates[1]))
    df = pd.concat([S0, S1])
    return df

def generate_sample_demographic_shift(data, rates, n_samples):
    '''
        - Simulate a demographic shift based on the sensitive values
        - Just P(S) changes 
        - Number of samples are fixed
    
    '''
    assert np.sum(rates) == 1.0
    S0 = data[data['sensitive'] == 0].sample(int(n_samples * rates[0]))
    S1 = data[data['sensitive'] == 1].sample(int(n_samples * rates[1]))
    df = pd.concat([S0, S1])
    return df


def generate_sample_joint_shift(data, rates, n_samples):
    
    '''
        Simulate a joint shift based on the sensitive and the y values together
    
    '''
    
    # assert np.sum(rates) <= 1.0
    # assert np.sum(rates) >= 0.9
    
    J00 = data.query('y == 0 and sensitive == 0').sample(int(n_samples * rates[0][0]), replace = True)
    J01 = data.query('y == 0 and sensitive == 1').sample(int(n_samples * rates[0][1]), replace = True)
    J10 = data.query('y == 1 and sensitive == 0').sample(int(n_samples * rates[1][0]), replace = True)
    J11 = data.query('y == 1 and sensitive == 1').sample(int(n_samples * rates[1][1]), replace = True)
    
    df = pd.concat([J00, J01, J10, J11])

    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def generate_spurious_correlation(df, train=False, test=None, alpha=0):
    
    if train:
        i = alpha
        p0 = np.array([[0.5,0],[0, 0.5]])
        p1 = np.array([[0, 0.5],[0.5,0]])
        rates = np.round_((1-i)*p0 + i*p1,2)
        train = generate_sample_joint_shift(df, rates, 5000)
        
        return train

    if test == 'balanced':
        lambdas = np.arange(0.1, 1, 0.1)
        i = 0.5
        p0 = np.array([[0.5,0],[0, 0.5]])
        p1 = np.array([[0, 0.5],[0.5,0]])
        list_tests = []
        for d in lambdas:
            rates = np.round_((1-i)*p0 + i*p1,2)
            test = generate_sample_joint_shift(df, rates, 5000)
            list_tests.append(test)
        
        return list_tests
    
    else:
        
        lambdas = np.arange(0.1, 1, 0.1)
        #lambdas = np.arange(0.1, 1, 0.01)
        #lambdas = lambdas[::5]
        p0 = np.array([[0.5,0],[0, 0.5]])
        p1 = np.array([[0, 0.5],[0.5,0]])

        list_tests = []
        for i in lambdas:
            rates = np.round_((1-i)*p0 + i*p1,2)
            test = generate_sample_joint_shift(df, rates, 5000)
            list_tests.append(test)
        
        return list_tests
    
def get_spurios_correlation_rate():
    all_rates = []
    lambdas = np.arange(0.1, 1, 0.1)
    # lambdas = np.arange(0.1, 1, 0.07)
    p0 = np.array([[0.5,0],[0, 0.5]])
    p1 = np.array([[0, 0.5],[0.5,0]])

    for i in lambdas:
        rates = np.round_((1-i)*p0 + i*p1,2)
        
        all_rates.append(rates)
        
    return all_rates


def get_subpopulation_shift():
    all_rates = []
    alphas = np.round(np.flip(np.arange(0.01, 0.26, 0.02)), 2)
    
    for alpha in alphas:
        base_joint = np.array([[alpha, 0.5 - alpha],[alpha, 0.5 - alpha]])
    
        all_rates.append(base_joint)
        
    return all_rates