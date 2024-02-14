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

agent = Agent(state_dim = 12, action_dim = 1, batch_size = 64) # Initialise the agent
np.random.seed(0) # enables consisent outputs from random number generation
""" agent.load_model_parameters() """ # load the trained model parameters for all the networks

## VC Coefficients
# VC Coefficients - general
Number_of_VCs = 100 # as a starting point 
Fund_maturity = 32 # number of time steps to realise returns (8 years) - each time step is 3 months (one quarter)

#################### CHECK VC QUALITY DISTRIBUTION (lognormal for TVPI), VC_QUALITY NORMALISATION, AND AVERAGE PORTFOLIO SIZE (AND ITS USE) ######################
VC_quality_shape = 0.39 # shape coefficient for lognormal distribution
VC_quality_loc = -0.49 # location coefficient for lognormal distribution
VC_max_scale = 1.70 # scale coefficient for lognormal distribution
VC_max_scale = 1 # we want to normalize VC_quality so that it lies between 0 and 1
Average_portfolio_size = 32 #Based on real world data

#################### CHANGE THIS ########################
#VC attributes - Employees
Number_of_employees_sd = 1.3711 #standard deviation coefficinet for lognormal distribution of number of employees
Number_of_employees_loc = 0.8426 #loc coefficient for lognormal distribution of number of employees
Number_of_employees_scale = 9.5626 #scale coefficient for lognormal distribution of number of employees
VC_work_hours_per_week = 56 #Average numebr of hours worked by an analyst in VC
Work_weeks_per_month = 4 
Work_hours_per_month = VC_work_hours_per_week*Work_weeks_per_month #Work hours per months per employee in VC
Work_hours_per_3months = Work_hours_per_month*3 #1 time step = 3 months, thus we are interested in hours per 3 months
Percentage_of_time_spend_on_other_activities = 0.31 #time spend by VC employee on activitties not related to either screening or advising
Time_for_screening_and_monitroring_3months_per_emp = Work_hours_per_3months*(1-Percentage_of_time_spend_on_other_activities)
Number_of_funds_at_operation = 2 #at same time, VC takes care of multiple funds
Time_for_screening_and_monitroring_3months_per_emp_per_fund = Time_for_screening_and_monitroring_3months_per_emp/Number_of_funds_at_operation
Average_number_of_investment_analysts = 19 #Based on real-world data

#################### CHANGE THIS #######################
#VC Coefficients - Time needed
Screening_time = 60 #Time in hours needed to screen a startup
Advising_time = 27.5 #Time in hours needed per time step(i.e. 3 months) to advise to a startup in the portfolio 

#################### CHECK NUMBER OF TIME STEPS TO EXIT ##############################
# VC Coefficients - Returns
VC_returns_alpha = 2.06 # alpha coefficient for power law distribution of VC retruns
VC_returns_x_min = 0 # X_min coefficeint for power law distribution of early stage returns
Startup_exit = 20 # number of time steps it takes a startup to exit (5 years)

#################### CHECK NUMBER OF NEW STARTUPS #########################
##Startup Coefficients
#Startup Coefficients - General
Number_of_new_startups = 25600 #Number of business starts in USA every 3 months, In fact, it is 256000, but for computational reasons we devide everything by 10
Growth_a = -2.89 # a parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
Growth_loc = 0.55 # loc parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
Growth_scale = 0.54 # scale parameter for the average skewed normal distribution of revenue growth for a startup, taken as a measure of potential
# dictionaries with loc and scale parameters for the revenue growth distribution for each subindustry
Sub_Industry_loc = {"Sub_Industry_1": 0.475, "Sub_Industry_2": 0.530, "Sub_Industry_3": 0.553, "Sub_Industry_4": 0.576, "Sub_Industry_5": 0.632}
Sub_Industry_scale = {"Sub_Industry_1": 0.466, "Sub_Industry_2": 0.520, "Sub_Industry_3": 0.543, "Sub_Industry_4": 0.565, "Sub_Industry_5": 0.621}

############### CHANGE THIS ###################
#Startup Coefficients - Time progression equation
#Gorwth = Alpha*Growth + Beta*VC + Idiosyncratic_shock
Alpha = 0.99 #alpha coefficient for time progression equation. Expresses weight of EPI
Beta = 0.01 #beta coefficient for time progression equation. Expresses the weight of VC 
Idiosyncratic_shock_mean = 0 #mean for normal distribution for idiosyncratic shock
Idiosyncratic_shock_sd = 0.0775 #standard deviation for normal distribtion for idiosyncratic shock

################## THINK OF MORE EFFICIENT WAY TO DO THIS - RANDOM CHOICE ###################
# Startup Coefficeints - Sub_Industries - same probability for each - random choice
List_of_Sub_Industries = ["Sub_Industry_1", "Sub_Industry_2", "Sub_Industry_3", "Sub_Industry_4", "Sub_Industry_5"]
Probability_Distribution_of_Sub_Industries = [0.2, 0.2, 0.2, 0.2, 0.2]

######################### COME UP WITH OWN COEFFICIENTS ############################
# Startup Coefficients - Due Diligence
Noise_mean_after_DD = 0 # mean for normal distribution of noise after due diligence
Noise_sd_after_DD = 0.25 # standard deviation for normal distribution of noise after due diligence

################# GET RID OF NEED OF MAX MUMBER, CHECK NUMBER OF DUE DILIGENCE INVESTORS ####################
# Startup Coefficients - Investors
Max_investments_per_startup = 1 # max number of investors allowed to invest in startup, 1 as each startup is in fact treated as an investment
Number_of_due_diligence_investors = 10 # number of investors enagaged in due diligence


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

#################### CHECK ESTIMATE OF SCREENINGS ######################
## General model coefficents
Risk_free_rate = 1.103 # Average of 10-Year US Treasury bill and 10-Year German government bond from 2008 to 2024, compounded 5 years
Estimate_of_screenings = int((Number_of_VCs * Average_number_of_investment_analysts * (Time_for_screening_and_monitroring_3months_per_emp_per_fund/Screening_time))/(Number_of_due_diligence_investors))

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
        ################# SEE HOW SCREENING PROSPECTS IS USED ###################
        self.Screening_prospects = []
        self.Portfolio = []
        self.Portfolio_size = len(self.Portfolio)

        self.Effort_left_for_fund = self.Investment_analysts*Time_for_screening_and_monitroring_3months_per_emp_per_fund
        self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time
        self.Effort_left_for_screening = self.Effort_left_for_fund - self.Effort_allocated_to_startups_in_portfolio
        self.Number_of_available_screenings = self.Effort_left_for_screening/Screening_time 
        self.Remaining_of_investment_stage = max(0, (Fund_maturity - Startup_exit - self.Fund_age)) # Investment stage is 3 years, so (8 years - 5 years - fund age)
    
        
    # This function enables us to map final revenue growth (startup potential) into returns
    def Growth_to_returns(self, growth):
        # This gives us probability of observing a growth less or equal to observed value - using the same probability distribution here (the average)
        Growth_cdf = skewnorm.cdf(growth, Growth_a, Growth_loc, Growth_scale)

        # return distribution of returns mapped from the final revenue growth data
        return float(sampled_VC_return_data[int(sample_size*Growth_cdf)])
    
    ######################### CHECK THIS AND WHAT TO INCLUDE #####################
    # Projects growth for startups into the future
    def projected_time_progression(self, growth):
        updated_growth = Alpha*growth + Beta*self.VC_quality + np.random.normal(Idiosyncratic_shock_mean, Idiosyncratic_shock_sd)
        return updated_growth
    

   ############################# THIS IS NOT USED #########################
    """                                 
    # Calculates the expected return of a Portfolio
    def expected_return(self, Portfolio):
        Return = 0
        # no return if there are no startups in the portfolio
        if len(Portfolio) == 0:
            return 0
        else:
            for i in Portfolio:
                # growth perceived by agent (VC)
                Perceived_Growth = getattr(i[0], "Growth_after_DD")
                for j in range(0,(Startup_exit-getattr(i[0],"Life_stage"))): 
                    # for each time step left for the startup, project its perceived growth                   
                    Projected_Growth = self.projected_time_progression(Perceived_Growth)
                    # Normalise so that there is no revenue growth of below -100% (impossible) or above 100% (too extreme - 256 multiple)
                    if Projected_Growth < -1:
                        Projected_Growth = -0.99
                    if Projected_Growth > 1:
                        Projected_Growth = 0.99
                
                ##################### CHECK WHY IS IT MULTIPLIED BY i[2] AND + RETURN - SEEMS LIKE i[2] IS ENDOWMENT/WEIGHT ASSIGNED TO STARTUP, BUT LINE 420 SUGGESTS OTHERWISE #####################
                Return = float((self.Growth_to_returns(Projected_Growth)*i[2])) + Return
            return Return
    
    """


    # Calculates the expected return without time projection - based only on the current perceived return
    def expected_return_without_projection(self, Portfolio):
        Return = 0
        if len(Portfolio) == 0:
            return 0
        else:
            for i in Portfolio:
                Projected_Growth = getattr(i[0], "Growth_after_DD")

                ################## CHECK THIS FORMULA AS FOR THE EXPECTED_RETURN FUNCTION ######################
                Return = float((self.Growth_to_returns(Projected_Growth)*i[2]))+ Return
            return Return



    # Calculate the final return when the portfolio is divested    
    def final_return(self, Portfolio):
        Return = 0
        for i in Portfolio:
            ############### CHECK i[2] HERE TOO - HERE IT SEEMS LIKE IT IS THE WEIGHT ##################
            Return = float((self.Growth_to_returns(getattr(i[0], "Growth"))*i[2]))+ Return
        return Return + self.Endowement


    def expected_portfolio_downside_deviation(self, Portfolio):
        sigma_d = 0
        for i in Portfolio:
            Projected_Growth = getattr(i[0], "Growth_after_DD")
            # calculate expected deviation from target return for each startup
            Dev = float(self.Growth_to_returns(Projected_Growth) - Risk_free_rate)
            Dev_squared = Dev**2
            # if the deviation is negative, sum the squared deviation
            if Dev < 0:
                sigma_d = sigma_d + Dev_squared       
        # calculate the expected downside deviation from the sum of the downside squared deviations
        sigma_d = sigma_d/(len(Portfolio) -1)
        sigma_d = sigma_d**(1/2)

        return sigma_d

            
    # Expected Sortino ratio
    def expected_Sortino_ratio(self, Portfolio):  
        if len(Portfolio) == 0:
            return 0
        else:
            ###################### WHY RETURN WITHOUT PROJECTION? ################################
            return float(self.expected_return_without_projection(Portfolio) - Risk_free_rate)/float(self.expected_portfolio_downside_deviation(Portfolio))
    

    # Returns 1 if a startup is in the portfolio, and 0 otherwise
    def startup_in_portfolio(self, Prospect):
        for i in self.Portfolio:
            if Prospect[0] in i:
                return 1
            else:
                return 0
        
    ################################ CHECK REWARDS ###############################
    # Gets reward after taking action a 
    def get_reward(self, action, startup):
        # Only can invest if within investment period
        if self.Fund_age <= (Fund_maturity - Startup_exit):  
            # Endowment cannot be negative 
            if action < 0:
                return torch.tensor([-100*(-action[0])])
            # If less than 0.005 was invested, we assume that VC does not invest into a given startup
            if 0<action <0.005:
                return torch.tensor([0])
            # If action is more than 0.005 but less than 1, then VC invests in startup
            if 0.005<=action<=1 and action <= self.Endowement:
                return torch.tensor([(self.expected_Sortino_ratio((self.Portfolio + [list(startup) + list(action)])) - self.expected_Sortino_ratio(self.Portfolio))])
            # If there is not enough endowment, no investment occurs
            if 0.005<=action<=1 and action > self.Endowement:
                return torch.tensor([-100*(action[0]-self.Endowement)])
            # Action cannot be more than 1
            if action>1:
                return torch.tensor([-100*action[0]])
        # No investment if investment period is past
        else:
            return torch.tensor([-10])


    # Gets state which is inputed into the RL model - this is what the agent observes
    def get_state(self, Prospect): 
        ## Prospect attributes
        # Attribtue 1 - prospect growth as perceived by agent (VC)
        Prospect_Growth = getattr(Prospect[0], "Growth_after_DD")

        ################ CHECK THIS TO SEE APPROPRIATE ATTRIBUTE FOR AGENT TO SEE - MAYBE SUBINDUSTRY STANDARD DEVIATION? #######################
        # Attribute 2 - prospect sub-industry
        Sub_Industry = getattr(Prospect[0], "Sub_Industry")[0]
        
        ################ SEE THIS AS THIS SEEMS TO BE FOR COHORT OF POSSIBLE SCREENINGS, NOT ACTUAL SCREENINGS ###################
        ## Cohort attributes
        # Attibute 3 - average growth of prospects, as perceived by agent (VC)
        total_cohort = 0
        for i in self.Screening_prospects:
            total_cohort = getattr(i[0], "Growth_after_DD") + total_cohort
        Screenings_mean = total_cohort/len(self.Screening_prospects) 
        # Attribute 4 - standard deviation of perceived growth of prospects by agent (VC)
        Screenings = []
        for i in self.Screening_prospects:
            Screenings.append(getattr(i[0], "Growth_after_DD"))
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
        # Attribute 7 - percentage of total screening capacity left, given a portfolio size
        Percentage_screening_left = self.Effort_left_for_screening/(Time_for_screening_and_monitroring_3months_per_emp_per_fund*self.Investment_analysts)
        # Attribute 8 - VC quality
        VC_quality = self.VC_quality
        # Attribute 9 - Endowment left (1 at the beginning)
        Endowement = self.Endowement
        # Attribute 10 - Remaining of investment stage as a percentage
        Remaining_of_investment_stage = Remaining_of_investment_stage/(Fund_maturity - Startup_exit)
        
        state_ = torch.tensor([Prospect_Growth, Sub_Industry, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Remaining_of_investment_stage])
        return state_
    
    # Gets next state 
    def get_next_state(self, action, Prospect):
        ################# I DO NOT GET WHY ACTION AND PROSPECT ARE NOT USED, AND WHY THESE ATTRIBUTES ARE ZERO - MAYBE BECAUSE IT IS NEXT STATE SO PROSPECTS FO NOT MATTER ANY MORE ####################
        ## Prospect attributes
        # Startup_in_portfolio = 0
        # Attribute 1 - prospect growth as perceived by agent (VC)
        Prospect_growth = 0

        ################ CHECK THIS TO SEE APPROPRIATE ATTRIBUTE FOR AGENT TO SEE - MAYBE SUBINDUSTRY STANDARD DEVIATION? #######################
        # Attribute 2 - prospect subindustry
        Sub_Industry = 0
        
        ################ SEE THIS AS THIS SEEMS TO BE FOR COHORT OF POSSIBLE SCREENINGS, NOT ACTUAL SCREENINGS ###################
        ## Cohort attributes
        # Attribute 3 - average growth of prospects, as perceived by agent (VC)
        total_cohort = 0
        for i in self.Screening_prospects:
            total_cohort = getattr(i[0], "Growth_after_DD") + total_cohort
        Screenings_mean = total_cohort/len(self.Screening_prospects) 
        # Attribute 4 - standard deviation of perceived growth of prospects by agent (VC)
        Screenings = []
        for i in self.Screening_prospects:
            Screenings.append(getattr(i[0], "Growth_after_DD"))
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
        EPIs = []
        Portfolio_sd = 0
        if self.Portfolio_size != 0:
            for i in self.Portfolio:
                EPIs.append(getattr(i[0], "Growth_after_DD"))
            Portfolio_sd = np.std(EPIs)
            
        ## VC attributes
        # Attribute 7 - percentage of total screening capacity left, given a portfolio size
        Percentage_screening_left = self.Effort_left_for_screening/(Time_for_screening_and_monitroring_3months_per_emp_per_fund*self.Investment_analysts)
        # Attribute 8 - VC quality
        VC_quality = self.VC_quality
        # Attribute 9 - Endowment left (1 at the beginning)
        Endowement = self.Endowement
        # Attribute 10 - Remaining of investment stage as a percentage
        Remaining_of_investment_stage = Remaining_of_investment_stage/(Fund_maturity - Startup_exit)
        
        next_state_ = torch.tensor([Prospect_growth, Sub_Industry, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Remaining_of_investment_stage])
        return next_state_
    
    # Executes the changes that occur at each time step
    def step(self):
        # VC only participates in matching and due diligence early on in their fund life cycle - during their investment stage
        if self.Fund_age <= (Fund_maturity - Startup_exit):
            for i in self.Screening_prospects:   
                    end = 0
                    obs = self.get_state(i) # observe next state
                    act = agent.select_action(obs) # select next action based on the observation of the next state
                    reward = self.get_reward(act, i) # a reward is given based on the action selected
                    if 0.005<=act<=1 and float(act) <= self.Endowement:
                        
                        ############################## WHY VC_INVESTMENTS? - SEEMS LIKE IT COULD BE THE VC INVESTMENTS IN A PARTICULAR STARTUP ################################
                        i[0].VC_investments.append(self)
                        self.Portfolio.append(i+[float(act)]+[float(getattr(i[0],"Growth_after_screening"))]+[float(self.Fund_age)]) # add the startup to the portfolio
                        self.Endowement = self.Endowement - float(act) # subtract action (investment) from endowment
                    new_state = self.get_next_state(act, obs) # get the next state based on the action and the observation
                    agent.remember(obs, act, reward, new_state, int(end)) # store the memory (s,a,r,s') in the replay buffer
                    agent.learn() # update network parameters
                    #agent.load_models()
                    #print(agent.memory.state_memory)
                    #print(agent.memory.action_memory)
                    #print("This is a sample")
                    #print(agent.memory.sample_buffer(1))
                    #print("This is a group of parameters")
                    #print(dict(agent.actor.named_parameters()))
                    print("This is the endowmwnet left")
                    print(self.Endowement) # print the endowment left after each action
                    print("This is the current portfolio size")
                    print(self.Portfolio_size) # print the portfolio size after each action
        self.Fund_age += 1 # update the age of the fund after each time step
        self.Portfolio_size = len(self.Portfolio)

        ############################ CHECK THIS ####################################
        self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time
        self.Effort_left_for_screening = self.Effort_left_for_fund - self.Effort_allocated_to_startups_in_portfolio
        self.Number_of_available_screenings = self.Effort_left_for_screening/Screening_time 

        self.Remaining_of_investment_stage = max(0, (Fund_maturity - Startup_exit - self.Fund_age)) # update the number of time steps left for the end of investment stage
        agent.save_model_parameters()




class Startup(Agent):
    def __init__(self, unique_id, Growth, Sub_Industry, Life_stage, model):
        self.unique_id = unique_id
        self.model = model
        self.Sub_Industry = Sub_Industry
        self.Growth = Growth
        self.Life_stage = Life_stage
        
        ##################### CHECK THIS ######################
        self.EPI_with_noise = 0

        self.Growth_after_DD = 0
        self.VC_potential_investments = []
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
        self.Growth = Alpha*self.Growth + Beta*self.average_investor_quality() + np.random.normal(Idiosyncratic_shock_mean, Idiosyncratic_shock_sd)
        self.Life_stage += 1    
        # Normalise so that there is no revenue growth of below -100% (impossible) or above 100% (too extreme - 256 multiple)
        if self.Growth < -1:
            self.Growth = -0.99
        if self.Growth > 1:
            self.Growth = 0.99


    ################################ CHECK HOW TO REMOVE THIS ###############################                                
    def noise_before_screening(self):
        if self.Life_stage == 0:
            self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
            while self.EPI_with_noise>1 or self.EPI_with_noise<0:
                self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
        else:
            self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES/(self.Life_stage**(1/2)))
            while self.EPI_with_noise>1 or self.EPI_with_noise<0:
                self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES/(self.Life_stage**(1/2)))
    
    def noise_after_screening(self):
        if self.Life_stage == 0:
            self.Growth_after_DD = self.Growth + np.random.normal(Noise_mean_after_DD, Noise_sd_after_DD)
            while self.Growth_after_DD>1 or self.Growth_after_DD<0:
                self.Growth_after_DD = self.Growth + np.random.normal(Noise_mean_after_DD, Noise_sd_after_DD)
        else:
            ################# CHECK IF LIFE STAGE IS YEARS OR QUARTERS, AS THIS WOULD HAVE AN INFLUENCE ON THE EQUATION BELOW FOR NOISE REDUCTION ###################
            self.Growth_after_DD = self.Growth + np.random.normal(Noise_mean_after_DD, Noise_sd_after_DD/(self.Life_stage**(1/2)))
            while self.Growth_after_DD>1 or self.Growth_after_DD<0:
                self.Growth_after_DD = self.Growth + np.random.normal(Noise_mean_after_DD, Noise_sd_after_DD)
    
    ############################# CHANGE THIS TO REMOVE THE NOISE BEFORE SCREENING #########################
    # Executes the changes that occur at each time step
    def step(self):
        #self.VC_potential_investments.sort(key = lambda x: x.VC_quality, reverse=True)
        #for i in self.VC_potential_investments[:5]:
            #self.VC_investments.append(i)
        #self.VC_potential_investments = []
        #Updating EPI with noise for startups
        self.noise_before_screening()
        self.noise_after_screening()
        #Collecting the prospects for this time step, 

        ############################ WHY THIS? #############################
        #0.450 and 0.570 correspond to levels of EPI that give return greater than 2
        if self.Life_stage == 0 and self.EPI_with_noise > 0.450:
            world.Early_stage_prospects.append(self)
        if self.Life_stage == 8 and self.EPI_with_noise > 0.570:
            world.Late_stage_prospects.append(self) 

        # We also make all the startups progress in time    
        self.time_progression()  
        
       
        
# Activation class, determines in which order agents are activated
class Activation_1(BaseScheduler):
    def step(self):
        # First, startups are activated
        for agent in self.agent_buffer(shuffled=True):
            if agent.unique_id >= world.VC_number:
                agent.step()
                
class Activation_2(BaseScheduler):
    def step(self):
        # Then, VCs are activated
        for agent in self.agent_buffer(shuffled=True):
            if agent.unique_id < world.VC_number:
                # After agents are activated, we update the step value
                agent.step()
        

class World(Model):
    def __init__(self, VC_number, Startup_number):
        self.VC_number = VC_number
        self.Startup_number = Startup_number
        # Schedules for the steps through the simulation
        self.schedule_1 = Activation_1(self) 
        self.schedule_2 = Activation_2(self)
        self.Early_stage_prospects = []
        self.Late_stage_prospects = []
        self.VCs = []
        
        # Creating Agents - VC
        for i in range (Number_of_VCs):

            ######################### CHANGE THIS WHEN NORMALISATION OF VC QUALITY IS WORKED OUT ############################
            a = VC(i,float(theoretical_distribution_VC.generate_random(1)/VC_max_TVPI),int(lognorm.rvs(Number_of_employees_sd, Number_of_employees_loc, Number_of_employees_scale)),0,self)
            while a.VC_quality>1:
                a.VC_quality = float(theoretical_distribution_VC.generate_random(1)/VC_max_TVPI)

            self.schedule_2.add(a)
            self.VCs.append(a)
        
        # Sort VCs in descending order of VC quality
        self.VCs.sort(key = attrgetter('VC_quality'), reverse = True)                
        
        # Collecting data
        self.datacollector = DataCollector(
          agent_reporters={"Growth": lambda a: getattr(a, "Growth", None)}
        )

    # Creating Agents - Startups_early stage
    def generate_startups(self):        
        for j in range (Number_of_VCs + self.schedule_1.steps*Number_of_new_startups, Number_of_VCs +  (self.schedule_1.steps+1)*Number_of_new_startups):
            # Get a sub-industry by random choice 
            Sub_Industry = random.choices(List_of_Sub_Industries, Probability_Distribution_of_Sub_Industries)
            # create the startup, and assign a revenue growth based on the distribution corresponding to the sub-industry
            b = Startup(j, skewnorm.rvs(Growth_a, Sub_Industry_loc[Sub_Industry], Sub_Industry_scale[Sub_Industry]), Sub_Industry, 0, self)
            # add the startup to the schedule
            self.schedule_1.add(b)         
       
            
    def matching_prospects_to_VCs(self):
        index = 0
        for i in world.Prospects:
            for j in world.VCs:

                ####################### CHECK THE CALCULATIONS FOR NUMBER OF AVAILABLE SCREENINGS ######################
                if getattr(j, "Number_of_available_screenings") > 1+ len(getattr(j, "Screening_prospects")):
                    j.Screening_prospects.append([i])

                    ############################# BETTER SEE HOW VC_POTENTIAL_INVESTMENTS WORKS #################
                    i.VC_potential_investments.append([j])

                    index +=1
                if index == 10:
                    break
            index = 0

        
    def step(self):
        self.Prospects = []
        self.generate_startups()
        self.schedule_1.step()

        ######################### CHECK EPI_WITH_NOISE #######################
        self.Prospects.sort(key = attrgetter('EPI_with_noise'), reverse = True)

        self.matching_prospects_to_VCs()
        self.schedule_2.step()
        self.schedule_1.steps += 1
        self.schedule_1.time += 1
        #self.datacollector.collect(self)

# Perform steps - simulation
world = World(Number_of_VCs, Number_of_new_startups)
for i in range(Fund_maturity):
    world.step()

# Get statistics on VC agents
Statistics_on_VCs = []
for i in world.VCs:
    Statistics_on_VCs = [[i.unique_id, i.VC_quality, i.Portfolio_size, i.Endowement, i.Investment_analysts, i.final_return(i.Portfolio)]] + Statistics_on_VCs

# Export the statistics on VC agents to an Excel spreadsheet
df = pd.DataFrame(np.array(Statistics_on_VCs), columns = ["Unique_id", "VC_Quality","Portfolio_size", "Endowement", "Investment_analysts", "Final_return"])
df.to_excel("Statistics_on_VCs.xlsx")
print(df)


Portfolio_data = []    
for i in world.VCs:
    for j in i.Portfolio:
        
        ############## CHECK IF THIS SHOULD BE Growth OR growth - THIS MAKES SENSE WITH LINE df2 BELOW - MEANS THAT FOR i in PORTFOLIO, i[0] IS STAGE, i[1] IS ENDOWMENT, i[2] IS GROWTH AFTER DD, AND i[3] IS FUND AGE #######################
        Transit = [i.unique_id, j[0].unique_id, j[0].Growth, i.Growth_to_returns(j[0].Growth, j[1])]

        ######################## SEE WHY j[0] IS NOT INCLUDED, AND WHY ONLY SUB-INDUSTRY FROM J[0]??? #######################
        # Access elements from j[1] onwards - do not include j[0]
        for z in j[1:]:
            Transit.append(z)
        Transit.append(j[0].Sub_Industry[0])

        Portfolio_data = [Transit] + Portfolio_data   

########################## CHECK THE COLUMNS ARE RIGHT, ESPECIALLY WITH "Stage" ######################
# Export the portfolio data to an Excel spreadsheet
df2 = pd.DataFrame(np.array(Portfolio_data), columns = ["Unique_id_VC", "Unique_id_Startup", "Growth_final", "Return", "Stage","Amount_Invested", "Growth_after_DD", "Fund_age", "Sub_Industry"])
df2.to_excel("Portfolio_data.xlsx")
print(df2) 
