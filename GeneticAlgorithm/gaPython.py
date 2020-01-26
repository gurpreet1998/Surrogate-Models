
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random
import math
import sys
import cmath
import meep as mp



ORDER = 0
SRC_POL = 1
result = pd.DataFrame(columns=["period", "height", "gap", "theta_1", "theta_2"])
global resolution
resolution = 50

def sim(queryrow, src_angle, nm_val, src_pol):
      global resolution
      result_df = pd.DataFrame(columns=result.columns)

      dpml = 1.0             # PML length
      dair = 4.0             # padding length between PML and grating
      dsub = 3.0             # substrate thickness
      d = queryrow["period"]           # grating period
      h = queryrow["height"]           # grating height
      g = queryrow["gap"]            # grating gap
      theta_1 = math.radians(queryrow["theta_1"])  # grating sidewall angle #1
      theta_2 = math.radians(queryrow["theta_2"])  # grating sidewall angle #2
      transmittance = 0

      sx = dpml+dair+h+dsub+dpml
      sy = d

      cell_size = mp.Vector3(sx,sy,0)
      pml_layers = [mp.Absorber(thickness=dpml,direction=mp.X)]

      wvl = 0.5              # center wavelength
      fcen = 1/wvl           # center frequency
      df = 0.05*fcen         # frequency width

      ng = 1.716             # episulfide refractive index @ 0.532 um
      glass = mp.Medium(index=ng)

      if src_pol == 1:
        src_cmpt = mp.Ez
        eig_parity = mp.ODD_Z
      elif src_pol == 2:
        src_cmpt = mp.Hz
        eig_parity = mp.EVEN_Z
      else:
        sys.exit("error: src_pol={} is invalid".format(args.src_pol))
        
      # rotation angle of incident planewave source; CCW about Z axis, 0 degrees along +X axis
      theta_src = math.radians(src_angle)
      
      # k (in source medium) with correct length (plane of incidence: XY)
      k = mp.Vector3(math.cos(theta_src),math.sin(theta_src),0).scale(fcen)
      if theta_src == 0:
        k = mp.Vector3(0,0,0)
      
      def pw_amp(k,x0):
        def _pw_amp(x):
          return cmath.exp(1j*2*math.pi*k.dot(x+x0))
        return _pw_amp

      src_pt = mp.Vector3(-0.5*sx+dpml+0.2*dair,0,0)
      sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                           component=src_cmpt,
                           center=src_pt,
                           size=mp.Vector3(0,sy,0),
                           amp_func=pw_amp(k,src_pt))]

      sim = mp.Simulation(resolution=resolution,
                          cell_size=cell_size,
                          boundary_layers=pml_layers,
                          k_point=k,
                          sources=sources)

      refl_pt = mp.Vector3(-0.5*sx+dpml+0.7*dair,0,0)
      refl_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0,sy,0)))

      sim.run(until_after_sources=100)

      input_flux = mp.get_fluxes(refl_flux)
      input_flux_data = sim.get_flux_data(refl_flux)

      sim.reset_meep()

      geometry = [mp.Block(material=glass, size=mp.Vector3(dpml+dsub,mp.inf,mp.inf), center=mp.Vector3(0.5*sx-0.5*(dpml+dsub),0,0)),
                  mp.Prism(material=glass,
                           height=mp.inf,
                           vertices=[mp.Vector3(0.5*sx-dpml-dsub,0.5*sy,0),
                                     mp.Vector3(0.5*sx-dpml-dsub-h,0.5*sy-h*math.tan(theta_2),0),
                                     mp.Vector3(0.5*sx-dpml-dsub-h,-0.5*sy+g-h*math.tan(theta_1),0),
                                     mp.Vector3(0.5*sx-dpml-dsub,-0.5*sy+g,0)])]

      sim = mp.Simulation(resolution=resolution,
                          cell_size=cell_size,
                          boundary_layers=pml_layers,
                          k_point=k,
                          sources=sources,
                          geometry=geometry)

      refl_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0,sy,0)))
      sim.load_minus_flux_data(refl_flux, input_flux_data)

      tran_pt = mp.Vector3(0.5*sx-dpml-0.5*dsub,0,0)
      tran_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0,sy,0)))

      sim.run(until_after_sources=500)

      kdom_tol = 1e-2
      angle_tol = 1e-6
      
      Rsum = 0
      Tsum = 0
      if theta_src == 0:
            nm_r = int(0.5*(np.floor((fcen-k.y)*d)-np.ceil((-fcen-k.y)*d)))       # number of reflected orders
            
            res = sim.get_eigenmode_coefficients(refl_flux, range(1,nm_r+1), eig_parity=eig_parity+mp.EVEN_Y)
            r_coeffs = res.alpha
            r_kdom = res.kdom
            for nm in range(nm_r):
              if nm != nm_val:
                  continue
              if r_kdom[nm].x > kdom_tol:
                r_angle = np.sign(r_kdom[nm].y)*math.acos(r_kdom[nm].x/fcen) if (r_kdom[nm].x % fcen > angle_tol) else 0
                Rmode = abs(r_coeffs[nm,0,1])**2/input_flux[0]
                print("refl: (even_y), {}, {:.2f}, {:.8f}".format(nm,math.degrees(r_angle),Rmode))
                Rsum += Rmode

            res = sim.get_eigenmode_coefficients(refl_flux, range(1,nm_r+1), eig_parity=eig_parity+mp.ODD_Y)
            r_coeffs = res.alpha
            r_kdom = res.kdom
            for nm in range(nm_r):
              if nm != nm_val:
                  continue
              if r_kdom[nm].x > kdom_tol:
                r_angle = np.sign(r_kdom[nm].y)*math.acos(r_kdom[nm].x/fcen) if (r_kdom[nm].x % fcen > angle_tol) else 0
                Rmode = abs(r_coeffs[nm,0,1])**2/input_flux[0]
                print("refl: (odd_y), {}, {:.2f}, {:.8f}".format(nm,math.degrees(r_angle),Rmode))
                Rsum += Rmode

            nm_t = int(0.5*(np.floor((fcen*ng-k.y)*d)-np.ceil((-fcen*ng-k.y)*d))) # number of transmitted orders

            res = sim.get_eigenmode_coefficients(tran_flux, range(1,nm_t+1), eig_parity=eig_parity+mp.EVEN_Y)
            t_coeffs = res.alpha
            t_kdom = res.kdom
            for nm in range(nm_t):
              if nm != nm_val:
                  continue
              if t_kdom[nm].x > kdom_tol:
                t_angle = np.sign(t_kdom[nm].y)*math.acos(t_kdom[nm].x/(ng*fcen)) if (t_kdom[nm].x % ng*fcen > angle_tol) else 0
                Tmode = abs(t_coeffs[nm,0,0])**2/input_flux[0]
                transmittance = transmittance + Tmode

                Tsum += Tmode

            res = sim.get_eigenmode_coefficients(tran_flux, range(1,nm_t+1), eig_parity=eig_parity+mp.ODD_Y)
            t_coeffs = res.alpha
            t_kdom = res.kdom
            for nm in range(nm_t):
              if nm != nm_val:
                  continue
              if t_kdom[nm].x > kdom_tol:
                t_angle = np.sign(t_kdom[nm].y)*math.acos(t_kdom[nm].x/(ng*fcen)) if (t_kdom[nm].x % ng*fcen > angle_tol) else 0
                Tmode = abs(t_coeffs[nm,0,0])**2/input_flux[0]
                transmittance = transmittance + Tmode

                Tsum += Tmode      
      else:
            nm_r = int(np.floor((fcen-k.y)*d)-np.ceil((-fcen-k.y)*d))       # number of reflected orders
            res = sim.get_eigenmode_coefficients(refl_flux, range(1,nm_r+1), eig_parity=eig_parity)
            r_coeffs = res.alpha
            r_kdom = res.kdom
            for nm in range(nm_r):
              if nm != nm_val:
                  continue
              if r_kdom[nm].x > kdom_tol:
                r_angle = np.sign(r_kdom[nm].y)*math.acos(r_kdom[nm].x/fcen) if (r_kdom[nm].x % fcen > angle_tol) else 0
                Rmode = abs(r_coeffs[nm,0,1])**2/input_flux[0]
                Rsum += Rmode

            nm_t = int(np.floor((fcen*ng-k.y)*d)-np.ceil((-fcen*ng-k.y)*d)) # number of transmitted orders
            res = sim.get_eigenmode_coefficients(tran_flux, range(1,nm_t+1), eig_parity=eig_parity)
            t_coeffs = res.alpha
            t_kdom = res.kdom
            for nm in range(nm_t):
              if nm != nm_val:
                  continue
              if t_kdom[nm].x > kdom_tol:
                t_angle = np.sign(t_kdom[nm].y)*math.acos(t_kdom[nm].x/(ng*fcen)) if (t_kdom[nm].x % ng*fcen > angle_tol) else 0
                Tmode = abs(t_coeffs[nm,0,0])**2/input_flux[0]
                transmittance = transmittance + Tmode

      return transmittance

def eval(chromosome):
    fitness = 0 
    angle_range = list(range(-45, 45, 5))
    for angle in angle_range:   # need to replace this with machine learning model
        transmittance = sim(chromosome, angle, ORDER, SRC_POL)
        fitness = fitness + transmittance
    return fitness

def crossover(parents):
    dice_xover = np.random.uniform(0,1)
    child = pd.DataFrame().reindex_like(parents)
    print('Crossover prob :', dice_xover)
    if dice_xover < 0.6:
        crossover_point = random.randint(1, len(parents.columns)-1)
        child.loc[0, :crossover_point] = parents.iloc[0, :crossover_point]
        child.loc[0, crossover_point:] = parents.iloc[1, crossover_point:]
        child.loc[1, :crossover_point] = parents.iloc[1, :crossover_point]
        child.loc[1, crossover_point:] = parents.iloc[0, crossover_point:]
    else:
        child.loc[0,:] = parents.loc[0,:]
        child.loc[1,:] = parents.loc[1,:]
    return child

def tournament_sel(population, SmallIsBetter):
    subpopulation = population.sample(n = int(len(population.index)/2))
    subpopulation = subpopulation.sort_values('Fitness', ascending=SmallIsBetter)
    subpopulation = subpopulation.reset_index(drop = True)
    print(subpopulation)
    parents = pd.DataFrame().reindex_like(population)
    parents.drop(parents.index, inplace=True)
    parents.drop(['Fitness'], axis='columns', inplace=True)
    print(parents)
    #assuming final column in population is fitness
    #loop to get parents
    for subpop in subpopulation.index:
        if len(parents) < 2:
        #loop to find exactly 2 parents to mate
            dice_tournament = np.random.uniform(0,1)
            if dice_tournament < 0.8:
                parents = parents.append(subpopulation.loc[subpop, :])
                parents = parents.reset_index(drop = True)
                continue
            else:
                continue
        else:
            break
    return parents
    
def mutate(child):
    nummutate = 0
    for chromosome in child.index:
        for gene in child.columns:
            dice_mutate = np.random.uniform(0,1)
            if dice_mutate < 0.2:
                val = child.loc[chromosome,gene] + np.random.uniform(-0.1,0.1)
                child.loc[chromosome,gene] = np.clip(val, 0, 1)
                nummutate = nummutate + 1
            else:
                continue
    print('nummutate = ', nummutate)
    return child




src_pol = 1
src_angle = 0
period = 0.6
gap = 0.1
height = 0.4
theta_1 = 12.8
theta_2 = 27.4
NUM_SAMPLE = 1000

def populate(row):
    row["period"] = (np.random.random() * 0.2  - 0.1) * period + period
    row["height"] = (np.random.random() * 0.2  - 0.1) * height +height
    row["gap"] = (np.random.random() * 0.2  - 0.1) * gap + gap
    row["theta_1"] = (np.random.random() * 0.2  - 0.1) * theta_1 + theta_1
    row["theta_2"] = (np.random.random() * 0.2  - 0.1) * theta_2 + theta_2

    return row
    
def optimize():
    popsize = NUM_SAMPLE
    numgenes = result.shape[1]
    
    df = pd.DataFrame(np.zeros((popsize, numgenes)), columns=result.columns)
    df = df.apply(populate, axis=1)

    print('New Population\n', df)
    num_generations = 50 # boleh change
    
    for generation in range(num_generations):
        print('Generation: ', generation)
        fitness = df.apply(lambda chr: eval(chr), 
                           axis=1)
        df['Fitness'] = fitness

        df_result = df.copy(deep = True)
        df_result = df_result.sort_values('Fitness', ascending=True)
        df_result = df_result.reset_index(drop=True)
        print('Generation ', generation, 'results:\n', df_result)
        df.drop(df.index, inplace = True)
        while(len(df.index) < len(df_result.index)):
            parents = tournament_sel(df_result, True)
            children = crossover(parents)
            mutated_child = mutate(children)
            df = df.append(mutated_child)
            df = df.reset_index(drop = True)
        #print('New Population\n', df)
    fitness = df.apply(lambda chr: eval(chr), axis=1)
    df_result = df.copy(deep=True)
    df_result['Fitness'] = fitness
    df_result = df_result.sort_values('Fitness',ascending=True)
    df_result = df_result.reset_index(drop=True)
    print('Best solution : ', df_result.loc[0, :])
    return df_result.loc[0,:]


optimize()
