#Libraries that used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def total_cost(dT):
    # Data
    df = pd.read_excel("heatForPython.xlsx")

    data_frame_cold = df[df['H/C'].str.contains('C')]
    data_frame_hot = df[df['H/C'].str.contains('H')]

    #End Points
    def endPointMaker(data, type='C'):
        temperatures = pd.concat([data['Tin'], data['Tout']])
        unique_sorted_temperatures = np.sort(temperatures.unique())
        dataFrame = pd.DataFrame(unique_sorted_temperatures, columns=[f'{type} end Points'])
        return dataFrame

    cold_end_point = endPointMaker(data_frame_cold, 'C')
    hot_end_point = endPointMaker(data_frame_hot, 'H')

    '''Pseudo end points'''
    pseudo_cold_end_points= cold_end_point + dT/2
    pseudo_hot_end_points= hot_end_point - dT/2


    def mergeAndSortEndpoints(cold_end_points, hot_end_points):
        merged = pd.concat([cold_end_points, hot_end_points]).reset_index(drop=True)
        merged = merged.drop_duplicates().sort_values(by=merged.columns[0]).reset_index(drop=True)
        combined_temperatures = pd.concat([merged['C end Points'], merged['H end Points']]).dropna()
        unique_sorted_temperatures = combined_temperatures.drop_duplicates().sort_values(ascending=False).reset_index(drop=True)
        return unique_sorted_temperatures

    '''Total End Points'''
    total_end_points = mergeAndSortEndpoints(cold_end_point, hot_end_point)
    '''Pseudo Total End Points'''
    total_pseudo_end_points = mergeAndSortEndpoints(pseudo_cold_end_points, pseudo_hot_end_points)

    # Phase change end point check
    def check_phase_change(df, temperature_list):
        # temperature_list'teki tüm değerleri float'a dönüştür
        temperature_list = [float(temp) for temp in temperature_list]  
        # 'phase change' değeri NaN olmayan satırları filtrele
        filtered_df = df.dropna(subset=['phase change'])
        # Filtrelenmiş satırları döngüye al
        for index, row in filtered_df.iterrows():
            tin = float(row['Tin'])
            tout = float(row['Tout'])        
            # Her iki değerin de listede olup olmadığını kontrol et
            if tin in temperature_list and tout in temperature_list:
                tin_index = temperature_list.index(tin)
                tout_index = temperature_list.index(tout)
            
                # Tin ve Tout'un listede sıralı olup olmadığını kontrol et
                if abs(tin_index - tout_index) == 1:
                    return True 
                else:
                    print(f"Tin ({tin}) ve Tout ({tout}) listede yan yana değil.")
            else:
                return False


    # Pinch Point,  Hot and Cold  Utulity
    # Data frame'e pseudo sıcaklıkları ekleme

    df['Pseudo Tin'] = df.apply(lambda row: row['Tin'] + dT/2 if 'C' in row['H/C'] else row['Tin'] - dT/2, axis=1)
    df['Pseudo Tout'] = df.apply(lambda row: row['Tout'] + dT/2 if 'C' in row['H/C'] else row['Tout'] - dT/2, axis=1)


    intervals = zip(total_pseudo_end_points, total_pseudo_end_points[1:])

    negative_dH = []

    # Aralıklar üzerinde döngü
    for upper, lower in intervals:
        '''ARALIKLAR'''
        # Mask kullanarak DataFrame'den verileri filtreleme
        # 'H' ile başlayanlar için mask
        mask_H = ((df['H/C'].str.startswith('H')) &
                ((df['Pseudo Tin'] >= upper) & (df['Pseudo Tout'] <= lower)))

        # 'C' ile başlayanlar için mask
        mask_C = ((df['H/C'].str.startswith('C')) &
                ((df['Pseudo Tin'] <= lower) & (df['Pseudo Tout'] >= upper)))

        # 'H' ile başlayanlar için seçilen akımlar
        selected_streams_H = df[mask_H]

        # 'C' ile başlayanlar için seçilen akımlar
        selected_streams_C = df[mask_C]
        selected_streams = pd.concat([selected_streams_H, selected_streams_C], ignore_index=True)

        '''-dH BULMAK İÇİN'''
        total_energy = 0
        for _, stream in selected_streams.iterrows():
            if stream['H/C'].startswith('H'):
                if pd.notna(stream['phase change']):
                    total_energy += stream['q (kj/s)']
                else:
                    total_energy += stream['nCp (kJ/s-K)'] * (upper - lower)
            else: 
                if pd.notna(stream['phase change']):
                    total_energy -= stream['q (kj/s)']
                else:
                    total_energy -= stream['nCp (kJ/s-K)'] * (upper - lower)
        negative_dH.append(total_energy)


    cumulative_energy_list= []
    energy_start=0
    for energy in negative_dH:
        energy_start += energy
        cumulative_energy_list.append(energy_start)

    min_hot_utulity= -min(cumulative_energy_list)


    cumulative_energy_list_start_with_hot_utility = []
    cumulative_energy_list_start_with_hot_utility.append(min_hot_utulity)
    energy_start=min_hot_utulity
    for energy in negative_dH:
        energy_start += energy
        cumulative_energy_list_start_with_hot_utility.append(energy_start)

    min_cold_utulity= cumulative_energy_list_start_with_hot_utility[-1]


    '''/////  Pinch Points /////'''
    # pinch point for hot
    pinch_indeks = np.abs(cumulative_energy_list_start_with_hot_utility).argmin()
    hot_pinch = total_pseudo_end_points [pinch_indeks] + dT/2
    # pinch point for cold
    cold_pinch = total_pseudo_end_points [pinch_indeks] - dT/2


    #print(f'Cold: Pinch : {cold_pinch}')
    #print(f'hot_pinch {hot_pinch}')

    '''HEN system'''
    split_fraction = 0.7

    exchanger_data = {
        'Heat exchanger' :  ["H1-C1", "H1-CU", "H2-CU", "H3-CU", "H2-C1/1", "H3-C1/2", "C1-HU", "C2-HU"],
        "H": ["H1", "H1", "H2", "H3", "H2", "H3","fh", "hps" ],
        "C": ["C1", "cw", "cw", "cw", "C1/1", "C1/2", "C1", "C2"],
        "nCp(h)": [float(df[df['H/C'].str.contains('H1')]['nCp (kJ/s-K)']), float(df[df['H/C'].str.contains('H1')]['nCp (kJ/s-K)']), float(df[df['H/C'].str.contains('H2')]['nCp (kJ/s-K)']), float(df[df['H/C'].str.contains('H3')]['nCp (kJ/s-K)']), float(df[df['H/C'].str.contains('H2')]['nCp (kJ/s-K)']), float(df[df['H/C'].str.contains('H3')]['nCp (kJ/s-K)']), np.nan, np.nan],
        "nCp (c)": [float(df[df['H/C'].str.contains('C1')]['nCp (kJ/s-K)']) , np.nan, np.nan, np.nan, float(df[df['H/C'].str.contains('C1')]['nCp (kJ/s-K)']) *split_fraction , float(df[df['H/C'].str.contains('C1')]['nCp (kJ/s-K)']) * (1- split_fraction) , float(df[df['H/C'].str.contains('C1')]['nCp (kJ/s-K)']) , np.nan],
        "T(h,in)": [hot_pinch, 'a' , hot_pinch, hot_pinch, float(df[df['H/C'].str.contains('H2')]['Tin']), float(df[df['H/C'].str.contains('H3')]['Tin']), 450, 370],
        "T(h,out)": ["a", float(df[df['H/C'].str.contains('H1')]['Tout']), float(df[df['H/C'].str.contains('H2')]['Tout']), float(df[df['H/C'].str.contains('H3')]['Tout']), hot_pinch,  hot_pinch, 350, 350],
        "T(c,in)": [float(df[df['H/C'].str.contains('C1')]['Tin']), 20, 20, 20, cold_pinch, cold_pinch, "#DEĞER!", float(df[df['H/C'].str.contains('C2')]['Tin'])],
        "T(c,out)": [cold_pinch, 45, 40, 30, "b", "c", float(df[df['H/C'].str.contains('C1')]['Tout']), float(df[df['H/C'].str.contains('C2')]['Tout'])],
        "Q": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", float(df[df['H/C'].str.contains('C2')]['q (kj/s)'])],
        'U' : [ (((427.8+2713.2)/(2*3600)) ),(((2713.2+13500)/(2*3600))) , (((1085.5+13500)/(2*3600))) , (((427.8+ 13500)/(2*3600)))  ,(((1085.5+2713.2)/(2*3600))) ,((( 427.8+2713.2)/(2*3600))) , (((2713.2+ 399.6)/(2*3600))) , (((775.8+21600)/(2*3600)))  ]
    }

    ''' Eksik dataların girişi'''
    exchanger_data['Q'][0] = (exchanger_data['T(c,out)'][0]  - exchanger_data['T(c,in)'][0]) * exchanger_data['nCp (c)'][0]
    exchanger_data['T(h,out)'][0] = (- exchanger_data['Q'][0])/ exchanger_data['nCp(h)'][0] + exchanger_data['T(h,in)'][0]
    exchanger_data['T(h,in)'][1]  = exchanger_data['T(h,out)'][0]
    exchanger_data['Q'][1] =  (exchanger_data['T(h,in)'][1]  - exchanger_data['T(h,out)'][1]) * exchanger_data['nCp(h)'][1]
    exchanger_data['Q'][2] = (exchanger_data['T(h,in)'][2]  - exchanger_data['T(h,out)'][2]) * exchanger_data['nCp(h)'][2]
    exchanger_data['Q'][3] = (exchanger_data['T(h,in)'][3]  - exchanger_data['T(h,out)'][3]) * exchanger_data['nCp(h)'][3]
    exchanger_data['Q'][4] = (exchanger_data['T(h,in)'][4]  - exchanger_data['T(h,out)'][4]) * exchanger_data['nCp(h)'][4]
    exchanger_data['T(c,out)'][4] = exchanger_data['Q'][4] / exchanger_data['nCp (c)'][4]  + exchanger_data['T(c,in)'][4]
    exchanger_data['Q'][5] = (exchanger_data['T(h,in)'][5]  - exchanger_data['T(h,out)'][5]) * exchanger_data['nCp(h)'][5]
    exchanger_data['T(c,out)'][5] = exchanger_data['Q'][5] / exchanger_data['nCp (c)'][5]  + exchanger_data['T(c,in)'][5]
    exchanger_data['T(c,in)'][6] = exchanger_data['T(c,out)'][4] * split_fraction + exchanger_data['T(c,out)'][5]* (1- split_fraction)
    exchanger_data['Q'][6] = (exchanger_data['T(c,out)'][6]  - exchanger_data['T(c,in)'][6]) * exchanger_data['nCp (c)'][6]

    exchanger_dataFrame = pd.DataFrame(exchanger_data)

    # dTlm sütununu hesapla ve veri çerçevesine ekle
    exchanger_dataFrame['dTlm'] = ((exchanger_dataFrame['T(h,in)'] - exchanger_dataFrame['T(c,out)']) - (exchanger_dataFrame['T(h,out)'] - exchanger_dataFrame['T(c,in)'])) / \
                (np.log((exchanger_dataFrame['T(h,in)'] - exchanger_dataFrame['T(c,out)']) / (exchanger_dataFrame['T(h,out)'] - exchanger_dataFrame['T(c,in)'])))
    exchanger_dataFrame['Area (m2)'] = exchanger_dataFrame['Q'] / (exchanger_dataFrame['U'] * exchanger_dataFrame['dTlm'])
    exchanger_dataFrame['Utility cp (kj/kgC)'] = pd.DataFrame([np.nan, 4.184, 4.184, 4.184 ,np.nan , np.nan , 1 , 1703 ])
    exchanger_dataFrame['Utility Flowrate (kg/h)'] = pd.DataFrame([ np.nan, 
                                                                (exchanger_dataFrame['Q'][1]/ (exchanger_dataFrame['Utility cp (kj/kgC)'][1] * (exchanger_dataFrame['T(c,out)'][1]- exchanger_dataFrame['T(c,in)'][1])))*3600, 
                                                                (exchanger_dataFrame['Q'][2]/ (exchanger_dataFrame['Utility cp (kj/kgC)'][2] * (exchanger_dataFrame['T(c,out)'][2]- exchanger_dataFrame['T(c,in)'][2])))*3600, 
                                                                (exchanger_dataFrame['Q'][3]/ (exchanger_dataFrame['Utility cp (kj/kgC)'][3] * (exchanger_dataFrame['T(c,out)'][3]- exchanger_dataFrame['T(c,in)'][3])))*3600,
                                                                np.nan ,
                                                                np.nan , 
                                                                (exchanger_dataFrame['Q'][6]/ (exchanger_dataFrame['Utility cp (kj/kgC)'][6] * (exchanger_dataFrame['T(h,in)'][6]- exchanger_dataFrame['T(h,out)'][6])))*3600, 
                                                                (exchanger_dataFrame['Q'][7]/ (exchanger_dataFrame['Utility cp (kj/kgC)'][7] * (exchanger_dataFrame['T(h,in)'][7]- exchanger_dataFrame['T(h,out)'][7])))*3600])

    cpi = 1.85
    life_time = 20 #year

    exchanger_dataFrame['Area cost ($/year)'] = ( (exchanger_dataFrame['Area (m2)'] ** 0.6) * (3* 10**3)* cpi ) / life_time

    cw_unit_cost = (0.08/1000)* cpi #  $/kg 
    fh_unit_cost = (0.35/(10**6))*cpi # $/kj
    hps_unit_cost = (4.4/1000) * cpi # $/kg
    
    exchanger_dataFrame['Utility Cost ($/Year)'] = pd.DataFrame([0,
                                                                exchanger_dataFrame['Utility Flowrate (kg/h)'][1]* cw_unit_cost*24*365,
                                                                exchanger_dataFrame['Utility Flowrate (kg/h)'][2]* cw_unit_cost*24*365,
                                                                exchanger_dataFrame['Utility Flowrate (kg/h)'][3]* cw_unit_cost*24*365,
                                                                0,
                                                                0,
                                                                exchanger_dataFrame['Q'][6]* fh_unit_cost * 3600*24*365,
                                                                exchanger_dataFrame['Utility Flowrate (kg/h)'][7]* hps_unit_cost*24*365])




    exchanger_dataFrame['Total Cost'] = exchanger_dataFrame['Area cost ($/year)'] + exchanger_dataFrame['Utility Cost ($/Year)']
    grand_total = sum(exchanger_dataFrame['Total Cost'])


    return grand_total



# dT değerlerini oluştur
dT_values = np.linspace(0.1, 30, 1000)  # 10 ile 30 arasında 100 adet değer

# Her bir dT için total_cost fonksiyonunu çağır ve sonuçları bir listeye kaydet
results = [total_cost(dT) for dT in dT_values]

print ( f'Optimum dT_min value is : {round(dT_values[np.argmin(results)],1)} degree Celsius.')
print ( f'Total annualized cost corresponding to that dT_min value is : {round(min(results)/1000000, 2)} M$/year')


# Sonuçları grafik üzerinde çiz
plt.plot(dT_values, results)
plt.xlabel('dT_min')
plt.ylabel('Total Cost ($/year)')
plt.title('Total Cost vs dT_min')
plt.show()