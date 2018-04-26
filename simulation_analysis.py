import pandas as pd
import graphs
import matplotlib.pyplot as plt
import os
import stats

# order_path = "/Users/jamesprince/project-data/orders.csv"
# trades_path = "/Users/jamesprince/project-data/trades.csv"

data_root = "/Users/jamesprince/project-data/sims/"
dirs = next(os.walk(data_root))[1]
print(dirs)

for dir in dirs[0]:
    orders_path = data_root + dir + "/orders.csv"
    trades_path = data_root + dir + "/trades.csv"

    # orders_df = pd.read_csv(order_path)
    # graphs.graph_order_sizes(orders_df)

    trades_df = pd.read_csv(trades_path)
    graphs.graph_price(trades_df)

    orders_df = pd.read_csv(orders_path)
    graphs.graph_price(orders_df)

    # graphs.graph_price_quantity(trades_df)


plt.show()



