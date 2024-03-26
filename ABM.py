# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:04:47 2024

@author: Gonzalo Plaza Molina
"""

from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from scipy.stats import skewnorm, lognorm
import numpy as np
import pandas as pd
import powerlaw as pr
from numpy.random import normal
import random
from operator import attrgetter
from SSTD3 import Agent, ReplayBuffer, OUNoise, Critic, Actor, Softmax
import torch

number_of_simulations = 50
Portfolio_size_data = []
Simulation_number = []
Statistics_on_VCs = []
Portfolio_data = []

for sim in range(number_of_simulations):

    agent = Agent(state_dim = 10, action_dim = 1) # Initialise the agent
    np.random.seed(0) # enables consisent outputs from random number generation
    if sim != 0:
        agent.load_model_parameters() # load the trained model parameters for all the networks
    print("Simulation", sim)

    ## VC Coefficients
    # VC Coefficients - general
    Number_of_VCs = 100 # 48,000 VCs in the world, but keeping it to 100 for computational reasons
    Fund_maturity = 40 # number of time steps to realise returns (10 years) - each time step is 3 months (one quarter)
    Startup_exit = 32 # number of time steps it takes a startup to exit (8 years)
    Average_portfolio_size = 32 #Based on real world data
    min_endowment_per_investment = 0.005 # Avoids the algorithm from making investments too small - limits max number of investments to 200 per fund (with noise accounted for, which sum to max 0.005)

    # VC Coefficients - VC quality and advisory
    VC_quality_shape = 0.385 # shape coefficient for lognormal distribution
    VC_quality_loc = -0.485 # location coefficient for lognormal distribution
    VC_quality_scale = 1.701 # scale coefficient for lognormal distribution
    VC_quality_max = 5.11 # 99.9% of the values for VC quality will be lower than 5.11, so it is used to normalise VC quality between 0 and 1.
    advisory_VC_quality_delta = 0.12 # this is done so that the advisory factor has the same mean as the the mean revenue growth rate (14.4%)


    # VC attributes - Employees
    # List and corresponding probabilities of number of investment professionals (here called analysts) in a VC firm - used for calculating total working hours per time step
    Number_of_analysts_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    Number_of_analysts_probabilities = [0.3042, 0.2195, 0.1531, 0.0966, 0.0649, 0.0430, 0.0282, 0.0223, 0.0156, 0.0126, 0.0083, 0.0066, 0.0041, 0.0038, 0.0029, 0.0024, 0.0019, 0.0017, 0.0012, 0.0011, 0.0012, 0.0012, 0.0010, 0.0012, 0.0011]
    Hours_per_week_analyst = 55 # Number of hours worked by an analyst in a VC firm
    Weeks_per_quarter = 13 # 52 weeks per year, so 13 weeks per quarter 
    Hours_per_quarter_analyst = Hours_per_week_analyst*Weeks_per_quarter # 1 time step = 3 months = 1 quarter
    Percentage_of_time_on_other_activities = 0.3 # Time spend by VC employee on activitties not related to either screening/DD or advising (e.g., fundraising)
    Time_for_DD_and_advising_per_analyst = Hours_per_quarter_analyst*(1-Percentage_of_time_on_other_activities) # per quarter
    Number_of_funds_per_VC = 1 # 89.5% of VC firms only manage one fund (an no firms manage more than 7), so we assume one fund for all VCs
    Time_for_DD_and_advising_per_analyst_per_fund = Time_for_DD_and_advising_per_analyst/Number_of_funds_per_VC # per quarter - this is the number used for later calculations

    # VC Coefficients - Time needed
    DD_time = 60 # Time in hours needed to perform due diligence a startup
    Advising_time = 45 # Time in hours needed per time step (i.e. 3 months) to advise to a startup in the portfolio 

    # VC Coefficients - Returns
    VC_returns_alpha = 2.06 # alpha coefficient for power law distribution of VC retruns
    VC_returns_x_min = 0 # X_min coefficeint for power law distribution of early stage returns

    ##Startup Coefficients
    #Startup Coefficients - General
    VCs_to_new_startups_ratio = 260 #The ratio of number of VCs to new startups every quarter worldwide
    Number_of_new_startups = Number_of_VCs*VCs_to_new_startups_ratio #Applies ratio to estimate the appropriate number of new startups per time step for the model
    Growth_a = -2.89 # a parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
    Growth_loc = 0.55 # loc parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
    Growth_scale = 0.54 # scale parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
    # dictionaries with loc and scale parameters for the revenue growth distribution for each subindustry
    Sub_Industry_loc = {"Sub_Industry_1": 0.475, "Sub_Industry_2": 0.530, "Sub_Industry_3": 0.553, "Sub_Industry_4": 0.576, "Sub_Industry_5": 0.632}
    Sub_Industry_scale = {"Sub_Industry_1": 0.466, "Sub_Industry_2": 0.520, "Sub_Industry_3": 0.543, "Sub_Industry_4": 0.565, "Sub_Industry_5": 0.621}

    # Startup Coefficients - Time progression equation
    # Gorwth = Alpha*Growth + Beta*Advisory + Idiosyncratic risk
    Beta = 0.00455 # beta coefficient for time progression equation. Expresses the weight of VC quality (advisory)
    Alpha = 1- Beta # alpha coefficient for time progression equation. Expresses weight of revenue growth

    Idiosyncratic_risk_mean = 0 # mean for normal distribution for idiosyncratic risk
    Idiosyncratic_risk_sd = 0.123 # standard deviation for normal distribution for idiosyncratic risk

    # Startup Coefficeints - Sub_Industries - same probability for each - random choice
    List_of_Sub_Industries = ["Sub_Industry_1", "Sub_Industry_2", "Sub_Industry_3", "Sub_Industry_4", "Sub_Industry_5"]
    Probability_Distribution_of_Sub_Industries = [0.2, 0.2, 0.2, 0.2, 0.2]
    # Matrix of correlations between subindustries
    Subindustry_correlation_matrix = pd.read_excel("Sub_Industry_Correlation_Matrix.xlsx")
    Subindustry_correlation_matrix.index = Subindustry_correlation_matrix["Unnamed: 0"]
    del Subindustry_correlation_matrix["Unnamed: 0"]

    # Startup Coefficients - Due Diligence
    Noise_mean_before_DD = 0 # mean for normal distribution of noise before due diligence
    # Standard deviation for normal distribution of noise before due diligence
    Noise_sd_before_DD = {"Sub_Industry_1": 0.307, "Sub_Industry_2": 0.342, "Sub_Industry_3": 0.357, "Sub_Industry_4": 0.372, "Sub_Industry_5": 0.407} 

    # Startup Coefficients - Investors
    Number_of_DD_investors = 5 # number of investors enagaged in due diligence per startup


    ## A sample of VC returns, used for mapping of revenue growth at exit -> to VC return
    # powerlaw package does not have an inverse cdf function, so we apporximate it with a sample
    sample_size = 10000
    VC_returns_distribution = pr.Power_Law(x_min = VC_returns_x_min, parameters = [VC_returns_alpha])
    simulated_data = VC_returns_distribution.generate_random(sample_size)
    simulated_data_new = []
    for i in simulated_data: # The power law starts with 1, so we have to shift everying to 0
        i = i-1
        simulated_data_new.append(i)
    sampled_VC_return_data = sorted(simulated_data_new)

    ## General model coefficents
    Risk_free_rate = 0.0198 # Average of 10-Year US Treasury bill and 10-Year German government bond from 2008 to 2024
    Compounded_risk_free_rate = (1+Risk_free_rate)**(Startup_exit/4) # Compounded 8 years and as a multiple

    ## Here we define class for VC, Startup and Activation
    # VC is assigend a unique id, VC quality and the number of investment analysts
    class VC(Agent):
        def __init__(self, unique_id, VC_quality, Investment_analysts, Fund_age, model):
            # Take inputs
            self.unique_id = unique_id
            self.model = model
            self.VC_quality = VC_quality
            self.Fund_age = Fund_age
            self.Investment_analysts = Investment_analysts

            
            self.Endowement = 1 # endowment normalised to 1
            self.Screening_prospects = []
            self.Portfolio = []
            self.Portfolio_size = len(self.Portfolio)

            self.Effort_available = self.Investment_analysts*Time_for_DD_and_advising_per_analyst_per_fund   # Total number of hours available for a fund per time step for DD and advising
            self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time # Effort allocated to advising startups currently in the portfolio
            self.Effort_left_for_DD = self.Effort_available - self.Effort_allocated_to_startups_in_portfolio # Effort left for DD after taking away the effort allocated to advisory
            self.Number_of_available_screenings = int((self.Effort_left_for_DD/DD_time)+1) # Number of possible screenings per time step based on the effort left for DD (rounded up)
            
            self.Remaining_of_investment_stage = max(0, (Fund_maturity - Startup_exit - self.Fund_age)) # Investment stage is 2 years, so (10 years - 8 years - fund age)
        
            
        # This function enables us to map revenue growth (startup potential) into returns
        def Growth_to_returns(self, growth):
            # This gives us probability of observing a growth less or equal to observed value - using the same probability distribution for all here (the average)
            Growth_cdf = skewnorm.cdf(growth, Growth_a, Growth_loc, Growth_scale)
            # distribution of returns mapped from the revenue growth data
            return float(sampled_VC_return_data[int(sample_size*Growth_cdf)])


        # Calculates the expected return without time projection - based only on the current perceived return
        def expected_return(self, Portfolio):
            Return = 0
            # No return if no startups in the portfolio
            if len(Portfolio) == 0:
                return 0
            else:
                # Get expected return based on the current revenue growth
                for i in Portfolio:
                    Projected_Growth = getattr(i[0], "Growth")

                    # i[1] would correspond to the action on the startup, that is the amount invested (so it is used as weight for portfolio return)
                    Return = float((self.Growth_to_returns(Projected_Growth)*i[1]))+ Return
                return Return


        # Calculate the actual return of the portfolio   
        def actual_return(self, Portfolio):
            Return = 0
            for i in Portfolio:
                # i[1] used as a weight here too
                Return = float((self.Growth_to_returns(getattr(i[0], "Growth")))*i[1])+ Return
            return Return + self.Endowement                                                                                     


        #Calculate expected variance for the whole portfolio 
        def expected_portfolio_variance(self, Portfolio):
            Total_variance = 0
            for i in Portfolio:
                    for j in Portfolio:
                        weight_i = i[1]/(1-self.Endowement)
                        weight_j = j[1]/(1-self.Endowement)
                        sd_i = (Noise_sd_before_DD[getattr(i[0], "Sub_Industry")]**2 + Idiosyncratic_risk_sd**2)**(1/2)
                        sd_j = (Noise_sd_before_DD[getattr(j[0], "Sub_Industry")]**2 + Idiosyncratic_risk_sd**2)**(1/2)
                        Correlation_coeff =  float(Subindustry_correlation_matrix.loc[getattr(i[0], "Sub_Industry"), getattr(j[0], "Sub_Industry")]) 
                        Total_variance = (weight_i*weight_j*Correlation_coeff*sd_i*sd_j) + Total_variance
            return Total_variance

            
        # Expected Sharpe ratio
        def expected_Sharpe_ratio(self, Portfolio):  
            if len(Portfolio) == 0:
                return 0
            else:
                return float(self.expected_return(Portfolio) - Compounded_risk_free_rate)/float(self.expected_portfolio_variance(Portfolio)**(1/2))


                # Gets reward after taking action a 
        def get_reward(self, action, old_portfolio):
            # Only can invest if within investment period
            if self.Fund_age <= (Fund_maturity - Startup_exit):  
                # Endowment cannot be negative 
                if action < 0:
                    return torch.tensor([-100*(-action[0])])
                # If less than the minimum endowment was invested, we assume that VC does not invest into a given startup
                if 0 <= action < min_endowment_per_investment:
                    return torch.tensor([0])
                # If action is more than the minimum endowment but less than 1, then VC invests in startup, and the reward is the Sortino ratio after minus Sortino ratio before
                if min_endowment_per_investment <= action <= 1 and action <= self.Endowement:
                    #return torch.tensor([(self.expected_Sortino_ratio((self.Portfolio + [list(startup) + list(action)])) - self.expected_Sortino_ratio(self.Portfolio))])
                    return torch.tensor([(self.expected_Sharpe_ratio(self.Portfolio) - self.expected_Sharpe_ratio(old_portfolio))])
                # If there is not enough endowment, no investment occurs
                if min_endowment_per_investment <= action <= 1 and action > self.Endowement:
                    return torch.tensor([-100*(action[0]-self.Endowement)])
                # Action cannot be more than 1
                if action>1:
                    return torch.tensor([-100*action[0]])
            # No investment if investment period is past - actually has no effect as no decisions are made once the investment stage is passed.
            else:
                return torch.tensor([0])
        

        # Gets state which is inputed into the RL model - this is what the agent observes
        def get_state(self, Prospect): 
            ## Prospect attributes
            # Attribtue 1 - prospect growth as perceived by agent (VC)
            Prospect_Growth = getattr(Prospect, "Growth_after_DD")
            # Attribute 2 - Average correlation of prospect with portfolio
            Avg_corr = 0
            for i in self.Portfolio:    
                Avg_corr =  float(Subindustry_correlation_matrix.loc[getattr(Prospect, "Sub_Industry"), getattr(i[0], "Sub_Industry")]) + Avg_corr
            if self.Portfolio_size != 0:
                Avg_corr = Avg_corr/self.Portfolio_size
            
            ## Cohort attributes
            # Attibute 3 - average growth of prospects, as perceived by agent (VC)
            total_cohort = 0
            for i in self.Screening_prospects:
                total_cohort = getattr(i, "Growth_after_DD") + total_cohort
            Screenings_mean = total_cohort/len(self.Screening_prospects) 
            # Attribute 4 - standard deviation of perceived growth of prospects by agent (VC)
            Screenings = []
            for i in self.Screening_prospects:
                Screenings.append(getattr(i, "Growth_after_DD"))
            Screenings_sd = np.std(Screenings)
                
            ## Portfolio attributes
            # Attribute 5 - mean perceived growth in portfolio by agent (VC)
            total = 0
            for i in self.Portfolio:
                total = getattr(i[0], "Growth_after_DD") + total
            Portfolio_mean = 0
            if self.Portfolio_size != 0:
                Portfolio_mean = total/self.Portfolio_size
            #Attribute 6 - standard deviation of perceived growth of portfolio companies by agent (VC)
            growths = []
            Portfolio_sd = 0
            if self.Portfolio_size != 0:
                for i in self.Portfolio:
                    growths.append(getattr(i[0], "Growth_after_DD"))
                Portfolio_sd = np.std(growths)
            
            ## VC attributes
            # Attribute 7 - percentage of total screening/DD capacity left, given a portfolio size
            Percentage_screening_left = self.Effort_left_for_DD/(Time_for_DD_and_advising_per_analyst_per_fund*self.Investment_analysts)
            # Attribute 8 - VC quality
            VC_quality = self.VC_quality
            # Attribute 9 - Endowment left (1 at the beginning)
            Endowement = self.Endowement
            # Attribute 10 - Remaining of investment stage as a percentage
            Remaining_of_investment_stage = self.Remaining_of_investment_stage/(Fund_maturity - Startup_exit)
            
            state_ = torch.tensor([Prospect_Growth, Avg_corr, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Remaining_of_investment_stage])
            return state_
        
        # Gets next state 
        def get_next_state(self, action, Prospect):
            ## Prospect attributes - no prospects on next state, so prospect attributes are null
            # Attribute 1 - prospect growth as perceived by agent (VC)
            Prospect_growth = 0
            # Attribute 2 - Average correlation of prospect with portfolio
            Avg_corr = 0
            
            ## Cohort attributes
            # Attribute 3 - average growth of prospects, as perceived by agent (VC)
            total_cohort = 0
            for i in self.Screening_prospects:
                total_cohort = getattr(i, "Growth_after_DD") + total_cohort
            Screenings_mean = total_cohort/len(self.Screening_prospects) 
            # Attribute 4 - standard deviation of perceived growth of prospects by agent (VC)
            Screenings = []
            for i in self.Screening_prospects:
                Screenings.append(getattr(i, "Growth_after_DD"))
            Screenings_sd = np.std(Screenings)
                
            # Portfolio attributes 
            # Attribute 5 - mean perceived growth in portfolio by agent (VC)
            total = 0
            for i in self.Portfolio:
                total = getattr(i[0], "Growth_after_DD") + total
            Portfolio_mean = 0
            if self.Portfolio_size != 0:
                Portfolio_mean = total/self.Portfolio_size
            # Attribute 6 - standard deviation of perceived growth of portfolio companies by agent (VC)
            growths = []
            Portfolio_sd = 0
            if self.Portfolio_size != 0:
                for i in self.Portfolio:
                    growths.append(getattr(i[0], "Growth_after_DD"))
                Portfolio_sd = np.std(growths)
            
            ## VC attributes
            # Attribute 7 - percentage of total screening capacity left, given a portfolio size
            Percentage_screening_left = self.Effort_left_for_DD/(Time_for_DD_and_advising_per_analyst_per_fund*self.Investment_analysts)
            # Attribute 8 - VC quality
            VC_quality = self.VC_quality
            # Attribute 9 - Endowment left (1 at the beginning)
            Endowement = self.Endowement
            # Attribute 10 - Remaining of investment stage as a percentage
            Remaining_of_investment_stage = self.Remaining_of_investment_stage/(Fund_maturity - Startup_exit)
            
            next_state_ = torch.tensor([Prospect_growth, Avg_corr, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Remaining_of_investment_stage])
            return next_state_
        
        # Executes the changes that occur at each time step
        def step(self):
            # VC only participates in matching and due diligence early on in their fund life cycle - during their investment stage
            Portfolio_temp = self.Portfolio
            if self.Fund_age <= (Fund_maturity - Startup_exit):
                for i in self.Screening_prospects: 
                        prospect_list = []  
                        end = 0
                        obs = self.get_state(i) # observe next state
                        act = agent.select_action(obs) # select next action based on the observation of the next state
                        if min_endowment_per_investment <= act <=1 and float(act) <= self.Endowement:
                            i.VC_investments.append(self) # Assign investor to startup
                            prospect_list.append(i)
                            self.Portfolio.append(prospect_list+[float(act)]+[float(getattr(i,"Growth_after_DD"))]+[float(self.Fund_age)]) # add the startup to the portfolio
                            self.Endowement = self.Endowement - float(act) # subtract action (investment) from endowment
                        reward = self.get_reward(act, Portfolio_temp) # a reward is given based on the action selected
                        new_state = self.get_next_state(act, obs) # get the next state based on the action and the observation
                        agent.remember(obs, act, reward, new_state, int(end)) # store the memory (s,a,r,s') in the replay buffer
                        agent.learn() # update network parameters
            self.Fund_age += 1 # update the age of the fund after each time step
            self.Portfolio_size = len(self.Portfolio)

            self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time
            self.Effort_left_for_DD = self.Effort_available - self.Effort_allocated_to_startups_in_portfolio
            self.Number_of_available_screenings = int((self.Effort_left_for_DD/DD_time)+1) # round up

            self.Remaining_of_investment_stage = max(0, (Fund_maturity - Startup_exit - self.Fund_age)) # update the number of time steps left for the end of investment stage
            agent.save_model_parameters()




    class Startup(Agent):
        def __init__(self, unique_id, Growth, Sub_Industry, Life_stage, model):
            self.unique_id = unique_id
            self.model = model
            self.Sub_Industry = Sub_Industry
            self.Growth = Growth
            self.Life_stage = Life_stage
            self.Growth_with_noise = 0
            self.Growth_after_DD = 0
            self.VC_investments = []
        
        # Calculate the average quality of the VC investing in a particular startup
        def average_investor_quality(self):
            total = 0
            if len(self.VC_investments) != 0:
                for i in self.VC_investments:
                    total = getattr(i, "VC_quality") + total
                return total/len(self.VC_investments)
            # if no VC has invested in a particular startup, then return zero
            if len(self.VC_investments) == 0:
                return 0
        

        # Startup progress in time                           
        def time_progression(self):
            # if no VC has ivnested in the startup yet, then progress without considering advisory form VC
            if len(self.VC_investments) == 0:
                self.Growth = self.Growth + np.random.normal(Idiosyncratic_risk_mean, Idiosyncratic_risk_sd)
            else:
                advisory_factor = self.average_investor_quality()-advisory_VC_quality_delta # VC quality distribution but same mean as the revenue growoth with good advisory
                self.Growth = Alpha*self.Growth + Beta*advisory_factor + np.random.normal(Idiosyncratic_risk_mean, Idiosyncratic_risk_sd)
            self.Life_stage += 1    
            # Normalise so that there is no revenue growth of below -100% (impossible) or above 100% equivalently - no extremes due to noise
            if self.Growth < -1:
                self.Growth = -1
            if self.Growth > 1:
                self.Growth = 1

        def noise_before_DD(self):
            # If the startup is new, the full noise is applied
            if self.Life_stage == 0:
                noise = np.random.normal(Noise_mean_before_DD, 2*Noise_sd_before_DD[self.Sub_Industry])
                self.Growth_with_noise = self.Growth + noise
                # Keep Growth with noise within limits
                while self.Growth_with_noise < -1:
                    self.Growth_with_noise = self.Growth_with_noise + abs(noise)
                while self.Growth_with_noise > 1:
                    self.Growth_with_noise = self.Growth_with_noise - abs(noise)
            # If the startup is not new, the noise is reduced in proportion to the maturity of the startup
            else:
                noise = np.random.normal(Noise_mean_before_DD, 2*Noise_sd_before_DD[self.Sub_Industry]/(self.Life_stage**(1/2)))
                self.Growth_with_noise = self.Growth + noise
                # Keep Growth with noise within limits
                while self.Growth_with_noise < -1:
                    self.Growth_with_noise = self.Growth_with_noise + abs(noise)
                while self.Growth_with_noise > 1:
                    self.Growth_with_noise = self.Growth_with_noise - abs(noise)


        def noise_after_DD(self):
            ## The following lines calculate and normalise VC quality, as it is used to reduce the noise for growth after DD
            # if no VC investors or potential investors yet, growth_with_noise is used as an approximation to VC Quality as the matching is done between these two - /
            # the cdf of the growth with noise is computed, then applied to the VC_quality distribution, and normalised.
            if len(self.VC_investments) == 0:
                VC_quality_cdf = skewnorm.cdf(self.Growth_with_noise, Growth_a, Growth_loc, Growth_scale)
                noise_reduction_VC_quality = lognorm.ppf(VC_quality_cdf, VC_quality_shape, VC_quality_loc, VC_quality_scale)/VC_quality_max
                # Ensure that the normalised VC quality for noise reduction is between 0 and 1 (not including, as too extreme/gives errors)
                if noise_reduction_VC_quality <= 0.01:
                    noise_reduction_VC_quality = 0.01
                if noise_reduction_VC_quality >= 0.99:
                    noise_reduction_VC_quality = 0.99
            # if there are VC investors, the average of the two means is taken
            else:
                noise_reduction_VC_quality = self.average_investor_quality()

            # 1 minus the VC quality for noise reduction is applied as a product of the standard deviation of the random noise, so the noise reduction is greater with a greater VC quality
            if self.Life_stage == 0:
                noise = np.random.normal(Noise_mean_before_DD, 2*Noise_sd_before_DD[self.Sub_Industry]*(1.5-noise_reduction_VC_quality))
                self.Growth_after_DD = self.Growth + noise
                # Keep perceived Growth within limits
                while self.Growth_after_DD < -1:
                    self.Growth_after_DD = self.Growth_after_DD + abs(noise)
                while self.Growth_after_DD > 1:
                    self.Growth_after_DD = self.Growth_after_DD - abs(noise)
            else: 
                noise = np.random.normal(Noise_mean_before_DD, 2*Noise_sd_before_DD[self.Sub_Industry]*(1.5-noise_reduction_VC_quality)/(self.Life_stage**(1/2)))
                self.Growth_after_DD = self.Growth + noise
                # Keep perceived Growth within limits
                while self.Growth_after_DD < -1:
                    self.Growth_after_DD = self.Growth_after_DD + abs(noise)
                while self.Growth_after_DD > 1:
                    self.Growth_after_DD = self.Growth_after_DD - abs(noise)
        
        # Executes the changes that occur at each time step
        def step(self):
            #Updating EPI with noise for startups
            self.noise_before_DD()
            self.noise_after_DD()
            #Collecting the prospects for this time step, 
            if self.Life_stage == 0 and world.counter < (Fund_maturity-Startup_exit):
                world.Prospects.append(self)
            # We also make all the startups progress in time    
            self.time_progression()  
            
        
            
    # Activation class, determines in which order agents are activated
    class Activation_1(BaseScheduler):
        def step(self):
            # Shuffle the agents to activate them in random order
            random.shuffle(self.agents)
            # First, startups are activated
            for agent in self.agents:
                agent.step()
                    
    class Activation_2(BaseScheduler):
        def step(self):
            # Shuffle the agents to activate them in random order
            random.shuffle(self.agents)
            # Then, VCs are activated
            for agent in self.agents:
                agent.step()
            

    class World(Model):
        def __init__(self, VC_number, Startup_number):
            self.VC_number = VC_number
            self.Startup_number = Startup_number
            # Schedules for the steps through the simulation
            self.schedule_1 = Activation_1(self) 
            self.schedule_2 = Activation_2(self)
            self.Prospects = []
            self.VCs = []
            self.counter = 0
            
            # Creating Agents - VC
            for i in range (Number_of_VCs):
                # for VC quality, draw a TVPI based on the TVPI distribution for VC quality and normalise
                Normalised_TVPI = lognorm.rvs(VC_quality_shape, VC_quality_loc, VC_quality_scale)/VC_quality_max
                # Ensure that the normalised VC quality for noise reduction is between 0 and 1:
                if Normalised_TVPI <= 0.01:
                    Normalised_TVPI = 0.01
                if Normalised_TVPI >= 0.99:
                    Normalised_TVPI = 0.99
                # Normalised_TVPI is passed as VC quality
                a = VC(i,float(Normalised_TVPI),random.choices(Number_of_analysts_list, Number_of_analysts_probabilities)[0],0,self)
                self.schedule_2.add(a)
                self.VCs.append(a)            
            
            # Collecting data
            self.datacollector = DataCollector(
            agent_reporters={"Growth": lambda a: getattr(a, "Growth", None)}
            )

        # Creating Agents - Startups
        def generate_startups(self):        
            for j in range (self.schedule_1.steps*Number_of_new_startups, (self.schedule_1.steps+1)*Number_of_new_startups):
                # Get a sub-industry by random choice 
                Sub_Industry = random.choices(List_of_Sub_Industries, Probability_Distribution_of_Sub_Industries)[0]
                # create the startup, and assign a revenue growth based on the distribution corresponding to the sub-industry
                b = Startup(j, skewnorm.rvs(Growth_a, Sub_Industry_loc[Sub_Industry], Sub_Industry_scale[Sub_Industry]), Sub_Industry, 0, self)
                # add the startup to the schedule
                self.schedule_1.add(b)         

                
        def matching_prospects_to_VCs(self):
            index = 0
            # clear list of screening prospects before adding new prospects 
            for k in world.VCs:
                k.Screening_prospects = []
            # Shuffle to go through prospects in a random order
            random.shuffle(world.Prospects)
            for i in world.Prospects:
                # Shuffle so that each prospect interacts with VCs randomly
                random.shuffle(world.VCs)
                for j in world.VCs:
                    if getattr(j, "Number_of_available_screenings") >= 1+ len(getattr(j, "Screening_prospects")):
                        j.Screening_prospects.append(i)
                        index += 1
                    # Ensures each prospect only gets the number of DD investors.
                    if index == Number_of_DD_investors:
                        break
                index = 0

            
        def step(self):
            # resets prospects list back to empty
            self.Prospects = []
            self.generate_startups()
            self.schedule_1.step()
            self.matching_prospects_to_VCs()
            self.schedule_2.step()

            # remove startups that are bankrupt - past their first quarter and not invested in
            to_remove = [startup for startup in self.schedule_1.agents if len(getattr(startup,"VC_investments")) == 0 and getattr(startup,"Life_stage") != 0]
            for startup in to_remove:
                self.schedule_1.remove(startup)

            self.schedule_1.steps += 1
            self.schedule_1.time += 1
            self.counter += 1
            print("Step",self.counter)
            print("Startups invested:", len(self.schedule_1.agents))
            print("Prospects:", len(self.Prospects))


    # Perform steps - simulation
    world = World(Number_of_VCs, Number_of_new_startups)
    for i in range(Fund_maturity):
        world.step()

    # Get statistics on VC agents
    for i in world.VCs:
        Statistics_on_VCs = [[sim, i.unique_id, i.VC_quality, i.Portfolio_size, i.Endowement, i.Investment_analysts, i.actual_return(i.Portfolio)]] + Statistics_on_VCs


    avg_portfolio_size = 0    
    for i in world.VCs:
        avg_portfolio_size = getattr(i, "Portfolio_size") + avg_portfolio_size
        for j in i.Portfolio:
            # Transit used to record all the data in a spreadsheet
            Transit = [sim, i.unique_id, j[0].unique_id, j[0].Growth, i.Growth_to_returns(j[0].Growth)]
            # Access elements from j[1] onwards - do not include j[0] as this is the startup itself (j[1] onwards includes amount invested, growth after DD, and fund age)
            for z in j[1:]:
                Transit.append(z)
            Transit.append(j[0].Sub_Industry)

            Portfolio_data = [Transit] + Portfolio_data   
    avg_portfolio_size = avg_portfolio_size/Number_of_VCs
    print(avg_portfolio_size)
    Portfolio_size_data.append(avg_portfolio_size)
    Simulation_number.append(sim)


# Export the statistics on VC agents to an Excel spreadsheet
df = pd.DataFrame(np.array(Statistics_on_VCs), columns = ["Simulation_number","Unique_id", "VC_Quality","Portfolio_size", "Endowement", "Investment_analysts", "Final_return"])
df.to_excel("Statistics_on_VCs.xlsx")

# Export the portfolio data to a spreadsheet
df2 = pd.DataFrame(np.array(Portfolio_data), columns = ["Simulation_Number","Unique_id_VC", "Unique_id_Startup", "Growth_final", "Return","Amount_Invested", "Growth_after_DD", "Fund_age_when_invested", "Sub_Industry"])
df2.to_excel("Portfolio_data.xlsx")

df3 = pd.DataFrame({"Simulation_Number": Simulation_number, "Average_Portfolio_Size": Portfolio_size_data})
df3.to_excel("Portfolio_size_data.xlsx", index=False)
print(df)
print(df2)
print(df3)
