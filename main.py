"""
Compute the BR for ALL/TRAIN and MSE/MAD, including a wrapper
"""

import logging
import collections
import numpy as np
from gurobipy import Model, GRB
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression as LReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.regression.quantile_regression import QuantReg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def statistics_dict():
    stats = collections.OrderedDict()
    # for reproducing
    stats['seed'] = None

    # Table 1 measurements
    stats['LS_TRAIN_Empirical'] = None
    stats['LS_TRAIN_True'] = None
    stats['LS_ALL_Empirical'] = None
    stats['LS_ALL_True'] = None
    stats['LA_TRAIN_Empirical'] = None
    stats['LA_TRAIN_True'] = None
    stats['LA_ALL_Empirical'] = None
    stats['LA_ALL_True'] = None

    # Table 2 measurements
    stats['LS_TRAIN_train_p1_MSE'] = None
    stats['LS_TRAIN_train_p1_MAE'] = None

    stats['LS_TRAIN_train_p2_MSE'] = None
    stats['LS_TRAIN_train_p2_MAE'] = None

    stats['LS_TRAIN_test_p1_MSE'] = None
    stats['LS_TRAIN_test_p1_MAE'] = None

    stats['LS_TRAIN_test_p2_MSE'] = None
    stats['LS_TRAIN_test_p2_MAE'] = None

    stats['LS_ALL_train_p1_MSE'] = None
    stats['LS_ALL_train_p1_MAE'] = None

    stats['LS_ALL_train_p2_MSE'] = None
    stats['LS_ALL_train_p2_MAE'] = None

    stats['LS_ALL_test_p1_MSE'] = None
    stats['LS_ALL_test_p1_MAE'] = None

    stats['LS_ALL_test_p2_MSE'] = None
    stats['LS_ALL_test_p2_MAE'] = None
    ##

    stats['LA_TRAIN_train_p1_MSE'] = None
    stats['LA_TRAIN_train_p1_MAE'] = None

    stats['LA_TRAIN_train_p2_MSE'] = None
    stats['LA_TRAIN_train_p2_MAE'] = None

    stats['LA_TRAIN_test_p1_MSE'] = None
    stats['LA_TRAIN_test_p1_MAE'] = None

    stats['LA_TRAIN_test_p2_MSE'] = None
    stats['LA_TRAIN_test_p2_MAE'] = None

    stats['LA_ALL_train_p1_MSE'] = None
    stats['LA_ALL_train_p1_MAE'] = None

    stats['LA_ALL_train_p2_MSE'] = None
    stats['LA_ALL_train_p2_MAE'] = None

    stats['LA_ALL_test_p1_MSE'] = None
    stats['LA_ALL_test_p1_MAE'] = None

    stats['LA_ALL_test_p2_MSE'] = None
    stats['LA_ALL_test_p2_MAE'] = None

    return stats


def best_response(X, Y, L, eps, B, timelimit):
    Gmodel = Model("mip1")
    m, d = X.shape
    Gmodel.params.OutputFlag = 0
    Gmodel.params.TimeLimit = timelimit
    logger.debug("Time Limit:{0}".format(Gmodel.params.TimeLimit))

    z = Gmodel.addVars(xrange(m), vtype=GRB.BINARY, name="z")
    h = Gmodel.addVars(xrange(d), vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name="h")
    h_intercept = Gmodel.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name="h_intercept")

    Gmodel.setObjective(sum([z[i] for i in xrange(m)]), GRB.MINIMIZE)
    reqCts = {}

    for i in xrange(m):
        reqCts[2 * i] = Gmodel.addConstr(
            Y[i] - sum([h[j] * X[i, j] for j in xrange(d)]) - h_intercept - B * z[i] <= L[i] - eps,
            "c_p_{0}".format(2 * i))
        reqCts[2 * i + 1] = Gmodel.addConstr(
            -Y[i] + sum([h[j] * X[i, j] for j in xrange(d)]) + h_intercept - B * z[i] <= L[i] - eps,
            "c_p_{0}".format(2 * i + 1))
    Gmodel.optimize()

    for v in Gmodel.getVars():
        logger.debug('%s %g' % (v.varName, v.x))

    logger.debug('Obj: %g' % Gmodel.objVal)

    h_star = [elem.x for elem in Gmodel.getVars()[m:-1]]
    h_intercept_star = Gmodel.getVars()[-1].x
    return h_star, h_intercept_star


def pff(y_hat1, y_hat2, Y):
    """
    Returns the proportion of points for which Player 2 is better predicting the true value
    :param y_hat1:
    :param y_hat2:
    :param Y:
    :return:
    """
    if type(Y) == type(list()):
        N = len(Y)
    elif type(Y) == type(np.array([])):
        N = Y.shape[0]

    L1 = abs(y_hat1 - Y)
    L2 = abs(y_hat2 - Y)
    return 1. * sum(L2 < L1) / N


def get_boston():
    from sklearn.datasets import load_boston
    boston = load_boston()
    return boston.get('data'), boston.get('target')


def instance(seed=None, test_size=0.2, eps=0.01, B=100., timelimit=60):
    X, Y = get_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    stats = statistics_dict()
    stats['seed'] = seed
    ########################### LS with TRAIN #############################
    p1model = LReg()
    p1model.fit(X_train, y_train)
    L = abs(p1model.predict(X_train) - y_train)
    h_star, h_intercept_star = best_response(X_train, y_train, L, eps, B, timelimit)
    y_hat1 = p1model.predict(X_train)
    y_hat2 = np.array([sum([h_star[j] * X_train[i, j] for j in xrange(X_train.shape[1])]) + h_intercept_star for i in
                       xrange(X_train.shape[0])])
    train_payoff = pff(y_hat1, y_hat2, y_train)
    logger.debug("LS,TRAIN:payoff of player 2 on the train set:{0}".format(train_payoff))

    stats['LS_TRAIN_train_p1_MSE'] = mean_squared_error(y_hat1, y_train)
    stats['LS_TRAIN_train_p1_MAE'] = mean_absolute_error(y_hat1, y_train)
    stats['LS_TRAIN_train_p2_MSE'] = mean_squared_error(y_hat2, y_train)
    stats['LS_TRAIN_train_p2_MAE'] = mean_absolute_error(y_hat2, y_train)

    del y_hat1, y_hat2

    y_hat1 = p1model.predict(X_test)
    y_hat2 = np.array([sum([h_star[j] * X_test[i, j] for j in xrange(X_test.shape[1])]) + h_intercept_star for i in
                       xrange(X_test.shape[0])])
    test_payoff = pff(y_hat1, y_hat2, y_test)
    logger.debug("LS,TRAIN:payoff of player 2 on the test set:{0}".format(test_payoff))

    stats['LS_TRAIN_test_p1_MSE'] = mean_squared_error(y_hat1, y_test)
    stats['LS_TRAIN_test_p1_MAE'] = mean_absolute_error(y_hat1, y_test)
    stats['LS_TRAIN_test_p2_MSE'] = mean_squared_error(y_hat2, y_test)
    stats['LS_TRAIN_test_p2_MAE'] = mean_absolute_error(y_hat2, y_test)
    del y_hat1, y_hat2, p1model

    stats['LS_TRAIN_Empirical'] = train_payoff
    stats['LS_TRAIN_True'] = test_payoff

    ########################### LS with ALL #############################
    p1model = LReg()
    p1model.fit(X, Y)
    L = abs(p1model.predict(X_train) - y_train)
    h_star, h_intercept_star = best_response(X_train, y_train, L, eps, B, timelimit)
    y_hat1 = p1model.predict(X_train)
    y_hat2 = np.array([sum([h_star[j] * X_train[i, j] for j in xrange(X_train.shape[1])]) + h_intercept_star for i in
                       xrange(X_train.shape[0])])
    train_payoff = pff(y_hat1, y_hat2, y_train)
    logger.debug("LS,ALL:payoff of player 2 on the train set:{0}".format(train_payoff))

    stats['LS_ALL_train_p1_MSE'] = mean_squared_error(y_hat1, y_train)
    stats['LS_ALL_train_p1_MAE'] = mean_absolute_error(y_hat1, y_train)
    stats['LS_ALL_train_p2_MSE'] = mean_squared_error(y_hat2, y_train)
    stats['LS_ALL_train_p2_MAE'] = mean_absolute_error(y_hat2, y_train)

    del y_hat1, y_hat2

    y_hat1 = p1model.predict(X_test)
    y_hat2 = np.array([sum([h_star[j] * X_test[i, j] for j in xrange(X_test.shape[1])]) + h_intercept_star for i in
                       xrange(X_test.shape[0])])
    test_payoff = pff(y_hat1, y_hat2, y_test)
    logger.debug("LS,ALL:payoff of player 2 on the test set:{0}".format(test_payoff))

    stats['LS_ALL_test_p1_MSE'] = mean_squared_error(y_hat1, y_test)
    stats['LS_ALL_test_p1_MAE'] = mean_absolute_error(y_hat1, y_test)
    stats['LS_ALL_test_p2_MSE'] = mean_squared_error(y_hat2, y_test)
    stats['LS_ALL_test_p2_MAE'] = mean_absolute_error(y_hat2, y_test)

    del y_hat1, y_hat2, p1model

    stats['LS_ALL_Empirical'] = train_payoff
    stats['LS_ALL_True'] = test_payoff
    ########################### LA with TRAIN #############################

    p1model = QuantReg(y_train, X_train)
    p1res = p1model.fit(q=.5)
    logger.debug(p1res.summary())
    logger.debug(p1res.params)
    prm = p1res.params
    L = abs(p1model.predict(params=prm, exog=X_train) - y_train)
    h_star, h_intercept_star = best_response(X_train, y_train, L, eps, B, timelimit)
    y_hat1 = p1model.predict(params=prm, exog=X_train)
    y_hat2 = np.array([sum([h_star[j] * X_train[i, j] for j in xrange(X_train.shape[1])]) + h_intercept_star for i in
                       xrange(X_train.shape[0])])
    train_payoff = pff(y_hat1, y_hat2, y_train)
    logger.debug("LA,TRAIN:payoff of player 2 on the train set:{0}".format(train_payoff))

    stats['LA_TRAIN_train_p1_MSE'] = mean_squared_error(y_hat1, y_train)
    stats['LA_TRAIN_train_p1_MAE'] = mean_absolute_error(y_hat1, y_train)
    stats['LA_TRAIN_train_p2_MSE'] = mean_squared_error(y_hat2, y_train)
    stats['LA_TRAIN_train_p2_MAE'] = mean_absolute_error(y_hat2, y_train)

    del y_hat1, y_hat2

    y_hat1 = p1model.predict(params=prm, exog=X_test)
    y_hat2 = np.array([sum([h_star[j] * X_test[i, j] for j in xrange(X_test.shape[1])]) + h_intercept_star for i in
                       xrange(X_test.shape[0])])
    test_payoff = pff(y_hat1, y_hat2, y_test)
    logger.debug("LA,TRAIN:payoff of player 2 on the test set:{0}".format(test_payoff))

    stats['LA_TRAIN_test_p1_MSE'] = mean_squared_error(y_hat1, y_test)
    stats['LA_TRAIN_test_p1_MAE'] = mean_absolute_error(y_hat1, y_test)
    stats['LA_TRAIN_test_p2_MSE'] = mean_squared_error(y_hat2, y_test)
    stats['LA_TRAIN_test_p2_MAE'] = mean_absolute_error(y_hat2, y_test)

    del y_hat1, y_hat2, p1model, p1res

    stats['LA_TRAIN_Empirical'] = train_payoff
    stats['LA_TRAIN_True'] = test_payoff
    ########################### LA with ALL #############################

    p1model = QuantReg(Y, X)
    p1res = p1model.fit(q=.5)
    logger.debug(p1res.summary())
    logger.debug(p1res.params)
    prm = p1res.params
    L = abs(p1model.predict(params=prm, exog=X_train) - y_train)
    h_star, h_intercept_star = best_response(X_train, y_train, L, eps, B, timelimit)
    y_hat1 = p1model.predict(params=prm, exog=X_train)
    y_hat2 = np.array([sum([h_star[j] * X_train[i, j] for j in xrange(X_train.shape[1])]) + h_intercept_star for i in
                       xrange(X_train.shape[0])])
    train_payoff = pff(y_hat1, y_hat2, y_train)
    logger.debug("LA,ALL:payoff of player 2 on the train set:{0}".format(train_payoff))

    stats['LA_ALL_train_p1_MSE'] = mean_squared_error(y_hat1, y_train)
    stats['LA_ALL_train_p1_MAE'] = mean_absolute_error(y_hat1, y_train)

    stats['LA_ALL_train_p2_MSE'] = mean_squared_error(y_hat2, y_train)
    stats['LA_ALL_train_p2_MAE'] = mean_absolute_error(y_hat2, y_train)

    del y_hat1, y_hat2

    y_hat1 = p1model.predict(params=prm, exog=X_test)
    y_hat2 = np.array([sum([h_star[j] * X_test[i, j] for j in xrange(X_test.shape[1])]) + h_intercept_star for i in
                       xrange(X_test.shape[0])])
    test_payoff = pff(y_hat1, y_hat2, y_test)
    logger.debug("LA,ALL:payoff of player 2 on the test set:{0}".format(test_payoff))

    stats['LA_ALL_test_p1_MSE'] = mean_squared_error(y_hat1, y_test)
    stats['LA_ALL_test_p1_MAE'] = mean_absolute_error(y_hat1, y_test)
    stats['LA_ALL_test_p2_MSE'] = mean_squared_error(y_hat2, y_test)
    stats['LA_ALL_test_p2_MAE'] = mean_absolute_error(y_hat2, y_test)

    del y_hat1, y_hat2, p1model

    stats['LA_ALL_Empirical'] = train_payoff
    stats['LA_ALL_True'] = test_payoff

    #############################  Finally  ##############################
    print stats
    line = ','.join([str(elm) for elm in stats.values()])
    return line


def loop(N=10):
    eps = 0.01
    test_size = 0.2
    B = 100.
    timelimit = 60
    np.random.seed(217)
    with open("PUBLISH-{0},{1}.csv".format(test_size, eps), 'a') as wfile:
        header = ",".join(statistics_dict().keys()) + '\n'
        wfile.writelines(header)
        for i in xrange(N):
            logger.info("working on the {0}'th loop".format(i + 1))
            seed = np.random.randint(1, 10000000)
            try:
                line = instance(seed=seed, test_size=test_size, eps=eps, B=B, timelimit=timelimit)
                print line
                wfile.writelines(line + '\n')
                wfile.flush()
            except Exception as e:
                print e
                continue


def wrap():
    loop(N=1000)


wrap()
