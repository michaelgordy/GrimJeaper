module GrimJeaper

export GrimModel, build_firm, build_estate, build_firmestate, 
       claimrecoveries, claimprices, 
       optimizekappa, equityFOC1, bankFOC1

using Distributions
using GSL   # or try SpecialFunctions
using Roots

Base.@kwdef struct GrimModel
   lambda::Real = 0.5
   r::Real = 0.03
   loanspread::Real = 0.01
   bondspread::Real = 0.02
   sigma::Real = 0.25
   delta::Real = 0
   rho::Real = 0.01 
   sigmatilde::Real = 0.2
   riskpremium::Real = 0
   loanaccrualspread::Real = 0 
   bondaccrualspread::Real = 0
   tau::Real = 1
   eta::Real = 0
   chi::Real = 0
   assetsales::Real = 1
   function GrimModel(lambda::Real, r::Real, 
                   loanspread::Real, bondspread::Real,
                   sigma::Real, 
                   delta::Real, rho::Real, 
                   sigmatilde::Real, riskpremium::Real,
                   loanaccrualspread::Real, bondaccrualspread::Real,
                   tau::Real, eta::Real, chi::Real,
                   assetsales::Real)
        (lambda>=0 && lambda<=1) || error("Must have lambda in [0,1]")
        (assetsales>=0 && assetsales<=1) || error("Must have assetsales in [0,1]")
        (rho>=0 && r>rho) || error("Must have r>rho>=0")
        (loanspread>0 && bondspread>=0) || error("Loan spread must be positive, bond spread nonnegative.")
        riskpremium>=0 || error("Risk premium in asset value must be nonnegative.")
        (loanaccrualspread>=0 && bondaccrualspread>=0) || error("Accrual spreads must be nonnegative.")
        (sigma>=0 && sigmatilde>=0) || error("Volatilities must be nonnegative.")
        (delta>=0 && tau>=0 && eta>=0) || error("Must have delta, tau, eta nonnegative")
        new(lambda, r, loanspread, bondspread, sigma, delta, rho,
            sigmatilde, riskpremium, loanaccrualspread, bondaccrualspread, tau, eta, chi, assetsales)
   end
end

# build_firm encapsulates model parameters for the living firm 
# Includes diffeq solution parameters as a list
function build_firm(gr0::GrimModel; lambda=gr0.lambda, 
                         loanspread=gr0.loanspread, 
                         bondspread=gr0.bondspread, 
                         r=gr0.r, sigma=gr0.sigma, 
                         assetsales=gr0.assetsales, 
                         delta=gr0.delta, rho=gr0.rho) 
  riskless = [lambda*(r+loanspread), (1-lambda)*(r+bondspread)]/r
  cashout = assetsales*(r + lambda*loanspread + (1-lambda)*bondspread) + delta 
  zeta = sigma^2/(2*cashout)
  bb = 1-2*(r-rho)/sigma^2
  alf = 0.5*(sqrt(bb^2+8*r/sigma^2)-bb)   # equation 20 in B-C
  bet = alf+1+bb
  return (lambda = lambda,
          loanspread = loanspread, bondspread = bondspread,
          riskless=riskless,
          assetsales=assetsales,
          r=r, sigma=sigma, delta=delta, rho=rho,
          alpha=alf, beta=bet, zeta=zeta, cashout=cashout)
end


# build_estate encapsulates model parameters for the estate of the bankrupt firm 
function build_estate(gr0::GrimModel; 
                      lambda=gr0.lambda, 
                      r=gr0.r, 
                      sigmatilde=gr0.sigmatilde, riskpremium=gr0.riskpremium,
                      loanaccrualspread=gr0.loanaccrualspread,
                      bondaccrualspread=gr0.bondaccrualspread,
                      tau=gr0.tau, eta=gr0.eta, chi=gr0.chi)  
  shockmean = exp(chi+eta^2/2)
  growthrateP = exp(tau*riskpremium)
  loanclaim = lambda*exp(tau*loanaccrualspread)
  debtclaim = loanclaim + (1-lambda)*exp(tau*bondaccrualspread)
  emergencevol = sqrt(tau*sigmatilde^2+eta^2)
  return (lambda=lambda, 
          loanclaim=loanclaim, debtclaim=debtclaim,
          shockmean=shockmean, growthrateP=growthrateP,
          emergencevol=emergencevol)
end

# Wrapper for build_firm and build_estate from parameter inputs
build_firmestate = function(gr0::GrimModel; 
                         lambda=gr0.lambda, r=gr0.r, 
                         loanspread=gr0.loanspread, 
                         bondspread=gr0.bondspread, 
                         sigma=gr0.sigma, 
                         delta=gr0.delta, rho=gr0.rho, 
                         sigmatilde=gr0.sigmatilde, riskpremium=gr0.riskpremium,
                         loanaccrualspread=gr0.loanaccrualspread, 
                         bondaccrualspread=gr0.bondaccrualspread, 
                         tau=gr0.tau, eta=gr0.eta, chi=gr0.chi,
                         assetsales=gr0.assetsales) 
   gr1 = GrimModel(lambda=lambda, r=r,
                 loanspread=loanspread, bondspread=bondspread,
                 sigma=sigma, 
                 delta=delta, rho=rho,  
                 sigmatilde=sigmatilde, riskpremium=riskpremium,
                 loanaccrualspread=loanaccrualspread, bondaccrualspread=bondaccrualspread,
                 tau=tau, eta=eta, chi=chi,
                 assetsales=assetsales)
   (build_firm(gr1), build_estate(gr1))
end

#mertonbond is the M function in paper.
#   This version assumes scalar inputs with D>0.
#   Handles V==0 without modification, but s==0 as special case.
function mertonbond(V,D,s)
  dplus  = log(V/D)/s + s/2
  dplus = ifelse(isnan(dplus),0,dplus)
  dminus = dplus - s 
  Phi = Normal()
  df = cdf(Phi,-dplus)
  M11 = -pdf(Phi, dplus)/(s*V)  
  dfds = -pdf(Phi,dplus)*V # M3
  ddfds = pdf(Phi, dplus)*dminus/s  # M13
  f  = V*df + D*cdf(Phi, dminus)
  return (M=f, M1=df, M2=cdf(Phi, dminus), 
                   M11=M11, M22=-pdf(Phi, dminus)/(s*D), 
                   M12=-V*M11/D, 
                   M3=dfds, M13=ddfds)
end

# psi and its derivatives are not vectorized
# Limiting case uses FWC 07.20.06.0009.01
function psi1(y, a, b, zeta, d::Int=0)
  if d==0
      Z = 1/(zeta*y)
      return y==0 ? exp(sf_lngamma(a+b)-sf_lngamma(b)) : Z^a * sf_hyperg_1F1(a,a+b,-Z)
  else
      return -a*zeta*psi1(y, a+1, b-1, zeta, d-1)
  end
end

# digital default option price.
#   Probably needs to be assigned notation for convenience, like A(V; kappa)..
function digitaldefaultprice(V, kappa, firm) 
  (V<kappa || kappa<0) && error("Require V geq kappa geq 0") 
  # deterministic case.  If nd[1]<0, default unattainable unless V==kappa
  V==kappa && return 1  
  if (firm.sigma==0) 
    nd = @. firm.cashout - (firm.r-firm.rho) * (V, kappa)
    ddp = nd[1]<=0 ? 0 : (nd[1]/nd[2])^(firm.r/(firm.r-firm.rho))
    return ddp
  end

  # zero cashflow taken from assets gives Leland 1994 boundary case
  firm.cashout==0 && return (kappa/V)^firm.alpha 

  # general case
  psi1(V, firm.alpha, firm.beta, firm.zeta) / 
    psi1(kappa, firm.alpha, firm.beta, firm.zeta)
end

# Define Xi1 as -1*D_V digitaldefaultoption evaluated at V=kappa 
#  Optional parameter m useless for calculating derivatives of Xi1.
function Xi1(kappa, firm, m::Int=0)
  m*(1-m)==0 || error("m must be 0 or 1") 
  # Deterministic case. Depends on whether net outflow of asset value is positive.
  #   Not sure if we are handling m correctly (by ignorning it) for this case.
  if firm.sigma==0 
    netoutflow = firm.cashout - (firm.r-firm.rho)*kappa
    return ifelse(netoutflow<=0, 0, firm.r/netoutflow) 
  end
  aa = firm.alpha+m
  bb = firm.beta-m
  # zero cashflow taken from assets gives Leland 1994 boundary case
  firm.cashout==0 && return aa/kappa 

  # general case
  -psi1(kappa, aa, bb, firm.zeta, 1) / 
      psi1(kappa, aa, bb, firm.zeta)
end

function dXi1(kappa, firm) 
  G0 = Xi1(kappa, firm)
  G0*(G0-Xi1(kappa, firm, m=1))
end

function claimprices(V, kappa, firm, chap11) 
  Vwhack = kappa*chap11.shockmean
  Mloan = mertonbond(Vwhack,chap11.loanclaim,chap11.emergencevol)
  # digital contingent claim on future bankruptcy event
  ddc = digitaldefaultprice(V,kappa,firm)
  Floan = firm.riskless[1]*(1-ddc) + Mloan.M*ddc
  Mtotal = mertonbond(Vwhack,chap11.debtclaim,chap11.emergencevol)
  Fbond = firm.riskless[2]*(1-ddc) + (Mtotal.M-Mloan.M)*ddc
  Fdeadweightloss = kappa*(1-chap11.shockmean)*ddc
  Fequity = V-Floan-Fbond-Fdeadweightloss
  return (loan=Floan, bond=Fbond, equity=Fequity,
               dwl=Fdeadweightloss)
end

# Recovery rates, market price and at emergence.  Returns list
function claimrecoveries(kappa,chap11) 
  Vwhack = kappa*chap11.shockmean
  # market prices at filing date
  Bloan = mertonbond(Vwhack,
                      chap11.loanclaim,
                      chap11.emergencevol).M
  Bdebt = mertonbond(Vwhack,
                       chap11.debtclaim,
                       chap11.emergencevol).M
  Bbond = Bdebt-Bloan
  Bequity = Vwhack - Bdebt
  # market prices at emergence
  Vemerge = Vwhack*chap11.growthrateP
  Eloan = mertonbond(Vemerge,
                      chap11.loanclaim,
                      chap11.emergencevol).M
  Edebt = mertonbond(Vemerge,
                      chap11.debtclaim,
                      chap11.emergencevol).M
  Ebond = Edebt-Eloan
  Eequity = Vemerge - Edebt
  return (debtRFV=Bdebt, loanRFV=Bloan, bondRFV=Bbond,
    equityRFV=Bequity,
    debtE=Edebt, loanE=Eloan, bondE=Ebond,
    equityE=Eequity)
end

# Probably want to replace this with comprehensive find_zeros() approach

# generic stepwise optimizer for bank and equity.
# slow, as it walks through the specified interval in small steps
function optimizekappa(focfunc; minkappa=0.0, maxkappa=2.0, dkappa=0.01) 
  kappa1 = minkappa
  reachpositive = reachnegative = false
  while kappa1 < maxkappa 
    reachpositive = (focfunc(kappa1)>0)
    ismissing(reachpositive) && return missing
    if reachpositive
        break
    end
    kappa1 += dkappa
  end
  reachpositive || return minkappa
  while kappa1 < maxkappa 
    reachnegative = focfunc(kappa1)<0
    ismissing(reachnegative) && return missing
    if (reachnegative) 
         break
    end   
    kappa1 += dkappa
  end
  reachnegative || return Inf
  find_zero(focfunc, (max(0,kappa1-2*dkappa), kappa1+dkappa), Bisection())
end

# Function to optimize for equity default threshold.
#  Assumes facevalue > 0
#  Originally negDFkappa1()
function equityFOC1(kappa, firm, chap11) 
  fkappa = mertonbond(kappa*chap11.shockmean,
                       chap11.debtclaim,
                       chap11.emergencevol)
  Bdebt = fkappa.M
  dBequity = chap11.shockmean*(1-fkappa.M1)  # marginal exercise value
  # now get marginal continuation value
  dE = 1 - (sum(firm.riskless) - Bdebt - 
               kappa*(1-chap11.shockmean))*Xi1(kappa, firm)
  return dBequity-dE
end

# Functions to optimize for bank foreclosure threshold.
#  Assumes facevalue > 0
function bankFOC1(kappa, firm, chap11)
  fkappa = mertonbond(kappa*chap11.shockmean,
                       chap11.loanclaim,
                       chap11.emergencevol)
  dB = chap11.shockmean*fkappa.M1
  dF = (firm.riskless[1] - fkappa.M)*Xi1(kappa, firm)
  return dB-dF
end
end # module
