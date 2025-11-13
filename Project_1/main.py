import pandas as pd
import matplotlib.pyplot as plt

dino_data = pd.read_csv('dnp_d.csv', nrows=1000)
dino_values = dino_data.loc[:, 'Zamkniecie'].tolist()
dino_dates = pd.to_datetime(dino_data.loc[:, 'Data'])

def ema(values, N):
    alfa = 2 / (N + 1)
    ema_values = [0] * len(values)
    ema_values[0] = values[0]

    for k in range(1, len(ema_values)):
        ema_values[k] = (alfa * values[k] + (1 - alfa) * ema_values[k - 1])
    
    return ema_values

def buy_sell(macd, signal):
    buy_signals = []
    sell_signals = []
    for k in range(len(macd)):
            if macd[k - 1] > signal[k - 1] and macd[k] < signal[k]:
                sell_signals.append(k)
            elif macd[k - 1] < signal[k - 1] and macd[k] > signal[k]:
                buy_signals.append(k)
    
    return buy_signals, sell_signals

def period_profit(buy_indexes, sell_indexes):
    profit = 0
    buy_prices = [dino_values[i] for i in buy_indexes]
    sell_prices = [dino_values[i] for i in sell_indexes]
    
    for i in range(len(buy_prices)):
        profit -= buy_prices[i]

    for i in range(len(sell_prices)):
        profit += sell_prices[i]
    
    return profit

def show_transactions_period(buy_indexes, sell_indexes):
    for i in range(len(buy_indexes)):
        buy_date = dino_dates[buy_indexes[i]].date()
        buy_price = dino_values[buy_indexes[i]]
        print(f"Kupno {buy_date} za {buy_price:.2f}")

    for i in range(len(sell_indexes)):
        sell_date = dino_dates[sell_indexes[i]].date()
        sell_price = dino_values[sell_indexes[i]]
        print(f"Sprzedaż: {sell_date} za {sell_price:.2f}")
    
    profit = period_profit(buy_indexes, sell_indexes)
    print(f"Zysk / Strata w wybranym okresie: {profit:.2f}")

def wallet_simulation():
    capital = 1000
    shares = 0
    wallet_values = []
    transactions = []
    buy_prices = []
    total_profit = 0
    max_value = 0
    min_value = 1000000000000

    for i in range(35, len(dino_values)):
        current_price = dino_values[i]
        current_date = dino_dates[i]
        wallet_value = capital + shares * current_price
        wallet_values.append(wallet_value)
        max_value = max(max_value, wallet_value)
        min_value = min(min_value, wallet_value)

        if i in buy_signals and capital >= current_price:
            # Kupujemy tyle akcji ile możemy
            max_shares = int(capital // current_price)
            if max_shares > 0:
                shares += max_shares
                capital -= max_shares * current_price
                buy_prices.extend([current_price] * max_shares)
                transactions.append(('KUP', current_date, current_price, max_shares, wallet_value))
        
        elif i in sell_signals and shares > 0:
            sell_price = shares * current_price
            capital += sell_price
            profit = sum(current_price - bp for bp in buy_prices)
            total_profit += profit
            transactions.append(('SPRZEDAJ', current_date, current_price, shares, wallet_value, profit))
            buy_prices = []
            shares = 0

    if shares > 0:
        final_price = dino_values[-1]
        final_date = dino_dates.iloc[-1]
        sell_price = shares * final_price
        capital += sell_price
        profit = sum(final_price - bp for bp in buy_prices)
        total_profit += profit
        wallet_value = capital
        wallet_values.append(wallet_value)
        transactions.append(('SPRZEDAJ', final_date, final_price, shares, wallet_value, profit))
        buy_prices = []
        shares = 0

    plt.figure(figsize=(15, 6))
    plt.plot(dino_dates[35:], wallet_values, label='Wartość portfela', color='blue', linewidth=2)
    plt.title(f"ZMIANY WARTOŚCI PORTFELA\n{transactions[0][1].strftime('%Y-%m-%d')} - {transactions[-1][1].strftime('%Y-%m-%d')}\n KAPITAŁ POCZĄTKOWY: 1000, KAPITAŁ KOŃCOWY: {wallet_values[-1]:.2f}") 
    plt.xlabel('DATA')
    plt.ylabel('WARTOŚĆ PORTFELA')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wykres_analiza_portfela.png", dpi=300)
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.plot(dino_dates[35:], wallet_values, label='WARTOŚĆ PORTFELA', color='blue', linewidth=2)
    for trans in transactions:
        if trans[0] == 'KUP':
            plt.scatter(trans[1], trans[4], color='green', marker='^', s=100, label='KUPNO' if trans == transactions[0] else "", zorder=3)
        else:
            plt.scatter(trans[1], trans[4], color='red', marker='v', s=100, label='SPRZEDAŻ' if trans == transactions[1] else "", zorder=3)
    plt.title(f"ZMIANY WARTOŚCI PORTFELA\n{transactions[0][1].strftime('%Y-%m-%d')} - {transactions[-1][1].strftime('%Y-%m-%d')}\n KAPITAŁ POCZĄTKOWY: 1000, KAPITAŁ KOŃCOWY: {wallet_values[-1]:.2f}") 
    plt.xlabel('DATA')
    plt.ylabel('WARTOŚĆ PORTFELA')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wykres_8_analiza_portfela_punkty_kupna_sprzedazy.png", dpi=300)
    plt.close()
    
    # Analiza transakcji
    profitable = sum(1 for t in transactions if t[0] == 'SPRZEDAJ' and t[5] > 0)
    unprofitable = sum(1 for t in transactions if t[0] == 'SPRZEDAJ' and t[5] <= 0)
    
    print(f"KAPITAŁ POCZĄTKOWY: 1000.00")
    print(f"KAPITAŁ KOŃCOWY: {wallet_values[-1]:.2f}")
    print(f"ZYSK CAŁKOWITY: {wallet_values[-1] - 1000:.2f}")
    print(f"NAJWIĘKSZA WARTOŚĆ PORTFELA W DANYM OKRESIE: {max_value:.2f}")
    print(f"NAJMNIEJSZA WARTOŚĆ PORTFELA W DANYM OKRESIE: {min_value:.2f}")
    print(f"LICZBA TRANSAKCJI KUPNA_SPRZEDAŻY: {len(transactions)//2}")
    print(f"TRANSAKCJE ZYSKOWNE: {profitable}")
    print(f"TRANSAKCJE STRATNE: {unprofitable}")
    print(f"SKUTECZNOŚĆ: {profitable/(profitable+unprofitable)*100:.1f}%")
    
    print("\nTRANSAKCJE:")
    for i, trans in enumerate(transactions):
        if trans[0] == 'KUP':
            print(f"TRANSAKCJA {i + 1}: {trans[1].strftime('%Y-%m-%d')} KUPNO {trans[3]} AKCJI PO {trans[2]:.2f}")
        else:
            print(f"TRANSAKCJA {i + 1}: {trans[1].strftime('%Y-%m-%d')} SPRZEDAŻ {trans[3]} AKCJI PO {trans[2]:.2f} (ZYSK: {trans[5]:.2f})")

ema_12 = ema(dino_values, 12)
ema_26 = ema(dino_values, 26)
macd = [(ema_12[k] - ema_26[k]) for k in range(len(dino_values))]
signal = ema(macd, 9)
buy_signals, sell_signals = buy_sell(macd, signal)

plt.figure(figsize=(12, 6))
plt.plot(dino_dates, dino_values, label='CENA ZAMKNIECIA', color = "black")
plt.title("WYKRES NOTOWANIA AKCJI SPÓŁKI DINO")
plt.xlabel('DATA')
plt.ylabel('CENA ZAMKNIECIA')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_1_cena_zamkniecia.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(dino_dates, dino_values, label='CENA ZAMKNIĘCIA', color='black')
plt.scatter(dino_dates.iloc[buy_signals], [dino_values[i] for i in buy_signals], color='green', marker='^', label='KUPNO', zorder=3)
plt.scatter(dino_dates.iloc[sell_signals], [dino_values[i] for i in sell_signals], color='orange', marker='v', label='SPRZEDAŻ', zorder=3)
plt.title("WYKRES NOTOWANIA AKCJI SPÓŁKI DINO Z ZAZNACZONYMI MIEJSCAMI KUPNA I SPRZEDAŻY")
plt.xlabel('DATA')
plt.ylabel('CENA ZAMKNIĘCIA')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_3_cena_zamkniecia.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(dino_dates, macd, label="MACD", color="blue")
plt.plot(dino_dates, signal, label="SIGNAL", color="red")
plt.title("MACD I SIGNAL")
plt.xlabel('DATA')
plt.ylabel('WARTOŚĆ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_macd_signal.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(dino_dates, macd, label="MACD", color="blue")
plt.plot(dino_dates, signal, label="SIGNAL", color="red")
plt.scatter(dino_dates.iloc[buy_signals], [macd[i] for i in buy_signals], color = 'green', marker = '^', label = "KUPNO", zorder = 3)
plt.scatter(dino_dates.iloc[sell_signals], [macd[i] for i in sell_signals], color = 'darkorange', marker = 'v', label = "SPRZEDAŻ", zorder = 3)
plt.title("MACD i SIGNAL Z PUNKTAMI KUPNA I SPRZEDAŻY")
plt.xlabel('DATA')
plt.ylabel('WARTOŚĆ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_2_macd_signal_punkty_kupna_sprzedazy.png", dpi=300)
plt.close()

range_start = 306
range_end = 420

period_dates = dino_dates[range_start:range_end]
period_values = dino_values[range_start:range_end]
period_macd = macd[range_start:range_end]
period_signal = signal[range_start:range_end]
all_buys = [i for i in buy_signals if range_start <= i < range_end]
period_buy = all_buys[:2]
all_sells = [i for i in sell_signals if range_start <= i < range_end]
period_sell = all_sells[:2]

show_transactions_period(period_buy, period_sell)

plt.figure(figsize=(12, 6))
plt.plot(period_dates, period_macd, label="MACD", color="blue")
plt.plot(period_dates, period_signal, label="SIGNAL", color="red")
plt.scatter(dino_dates.iloc[period_buy], [macd[i] for i in period_buy], color='green', marker='^', label='KUPNO', s=100, zorder=3)
plt.scatter(dino_dates.iloc[period_sell], [macd[i] for i in period_sell], color='orange', marker='v', label='SPRZEDAŻ', s=100, zorder=3)
plt.title("FRAGMENT ANALIZY MACD I SIGNAL")
plt.xlabel("DATA")
plt.ylabel("WARTOŚĆ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_4_analiza_fragmentu_macd_signal.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(period_dates, period_values, label="CENA ZAMKNIĘCIA", color="black")
plt.scatter(dino_dates.iloc[period_buy], [dino_values[i] for i in period_buy], color='green', marker='^', s=100, label='KUPNO', zorder=3)
plt.scatter(dino_dates.iloc[period_sell], [dino_values[i] for i in period_sell], color='orange', marker='v', s=100, label='SPRZEDAŻ', zorder=3)
plt.title("FRAGMENT NOTOWAŃ AKCJI SPÓŁKI DINO Z PUNKTAMI KUPNA I SPRZEDAŻY")
plt.xlabel("DATA")
plt.ylabel("CENA ZAMKNIĘCIA")
plt.grid(True)
plt.tight_layout()
plt.savefig("wykres_5_analiza_fragmentu_notowania.png", dpi=300)
plt.close()

range_start = 565
range_end = 660

period_dates = dino_dates[range_start:range_end]
period_values = dino_values[range_start:range_end]
period_macd = macd[range_start:range_end]
period_signal = signal[range_start:range_end]
all_buys = [i for i in buy_signals if range_start <= i < range_end]
period_buy = all_buys[:2]
all_sells = [i for i in sell_signals if range_start <= i < range_end]
period_sell = all_sells[:2]

show_transactions_period(period_buy, period_sell)

plt.figure(figsize=(12, 6))
plt.plot(period_dates, period_macd, label="MACD", color="blue")
plt.plot(period_dates, period_signal, label="SIGNAL", color="red")
plt.scatter(dino_dates.iloc[period_buy], [macd[i] for i in period_buy], color='green', marker='^', label='KUPNO', s=100, zorder=3)
plt.scatter(dino_dates.iloc[period_sell], [macd[i] for i in period_sell], color='orange', marker='v', label='SPRZEDAŻ', s=100, zorder=3)
plt.title("FRAGMENT ANALIZY MACD I SIGNAL")
plt.xlabel("DATA")
plt.ylabel("WARTOŚĆ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_6_analiza_fragmentu_macd_signal.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(period_dates, period_values, label="CENA ZAMKNIĘCIA", color="black")
plt.scatter(dino_dates.iloc[period_buy], [dino_values[i] for i in period_buy], color='green', marker='^', s=100, label='KUPNO', zorder=3)
plt.scatter(dino_dates.iloc[period_sell], [dino_values[i] for i in period_sell], color='orange', marker='v', s=100, label='SPRZEDAŻ', zorder=3)
plt.title("FRAGMENT NOTOWAŃ AKCJI SPÓŁKI DINO Z PUNKTAMI KUPNA I SPRZEDAŻY")
plt.xlabel("DATA")
plt.ylabel("CENA ZAMKNIĘCIA")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wykres_7_analiza_fragmentu_notowania.png", dpi=300)
plt.close()

wallet_simulation()