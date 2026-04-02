"""
=============================================================================
SISTEMA DE CONTROL DIFUSO PARA REGULACIÓN TÉRMICA
Universidad de Cartagena - Facultad de Ingeniería
Programa Ingeniería de Sistemas - Inteligencia Artificial
Grupo 1 - Lógica Difusa

Elba Puello Pérez
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSOS DE DISCURSO
# ─────────────────────────────────────────────────────────────────────────────
temperatura_range  = np.arange(10, 30.1, 0.1)   # °C  [10, 30]
tasa_cambio_range  = np.arange(-3, 3.1, 0.1)    # °C/min  [-3, 3]
potencia_range     = np.arange(-100, 100.1, 0.5) # %  [-100, 100]

# ─────────────────────────────────────────────────────────────────────────────
# 2. VARIABLES DIFUSAS (Antecedentes y Consecuente)
# ─────────────────────────────────────────────────────────────────────────────
temperatura  = ctrl.Antecedent(temperatura_range,  'temperatura')
tasa_cambio  = ctrl.Antecedent(tasa_cambio_range,  'tasa_cambio')
potencia     = ctrl.Consequent(potencia_range,     'potencia',   defuzzify_method='centroid')

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONJUNTOS DIFUSOS
# ─────────────────────────────────────────────────────────────────────────────

# --- Temperatura (5 conjuntos) ---
# MuyFria: Trapezoidal [10, 10, 13, 16]
temperatura['MuyFria']     = fuzz.trapmf(temperatura_range, [10, 10, 13, 16])
# Fria: Triangular [13, 16, 19]
temperatura['Fria']        = fuzz.trimf(temperatura_range,  [13, 16, 19])
# Confort: Gaussiana μ=20, σ=1
temperatura['Confort']     = fuzz.gaussmf(temperatura_range, 20, 1)
# Caliente: Triangular [19, 22, 25]
temperatura['Caliente']    = fuzz.trimf(temperatura_range,  [19, 22, 25])
# MuyCaliente: Trapezoidal [24, 27, 30, 30]
temperatura['MuyCaliente'] = fuzz.trapmf(temperatura_range, [24, 27, 30, 30])

# --- Tasa de Cambio (3 conjuntos) ---
# Bajando: Trapezoidal [-3, -3, -1, 0]
tasa_cambio['Bajando']  = fuzz.trapmf(tasa_cambio_range, [-3, -3, -1, 0])
# Estable: Gaussiana μ=0, σ=0.5
tasa_cambio['Estable']  = fuzz.gaussmf(tasa_cambio_range, 0, 0.5)
# Subiendo: Trapezoidal [0, 1, 3, 3]
tasa_cambio['Subiendo'] = fuzz.trapmf(tasa_cambio_range, [0, 1, 3, 3])

# --- Potencia de Control (5 conjuntos) ---
# EnfriarMucho: Trapezoidal [-100, -100, -70, -40]
potencia['EnfriarMucho']  = fuzz.trapmf(potencia_range, [-100, -100, -70, -40])
# EnfriarPoco: Triangular [-60, -30, 0]
potencia['EnfriarPoco']   = fuzz.trimf(potencia_range,  [-60, -30, 0])
# Neutro: Gaussiana μ=0, σ=10
potencia['Neutro']        = fuzz.gaussmf(potencia_range, 0, 10)
# CalentarPoco: Triangular [0, 30, 60]
potencia['CalentarPoco']  = fuzz.trimf(potencia_range,  [0, 30, 60])
# CalentarMucho: Trapezoidal [40, 70, 100, 100]
potencia['CalentarMucho'] = fuzz.trapmf(potencia_range, [40, 70, 100, 100])

# ─────────────────────────────────────────────────────────────────────────────
# 4. BASE DE REGLAS (15 reglas)
# ─────────────────────────────────────────────────────────────────────────────
reglas = [
    # Temperatura MuyFria
    ctrl.Rule(temperatura['MuyFria']     & tasa_cambio['Bajando'],  potencia['CalentarMucho']),  # R1
    ctrl.Rule(temperatura['MuyFria']     & tasa_cambio['Estable'],  potencia['CalentarMucho']),  # R2
    ctrl.Rule(temperatura['MuyFria']     & tasa_cambio['Subiendo'], potencia['CalentarPoco']),   # R3
    # Temperatura Fria
    ctrl.Rule(temperatura['Fria']        & tasa_cambio['Bajando'],  potencia['CalentarMucho']),  # R4
    ctrl.Rule(temperatura['Fria']        & tasa_cambio['Estable'],  potencia['CalentarPoco']),   # R5
    ctrl.Rule(temperatura['Fria']        & tasa_cambio['Subiendo'], potencia['Neutro']),          # R6
    # Temperatura Confort
    ctrl.Rule(temperatura['Confort']     & tasa_cambio['Bajando'],  potencia['CalentarPoco']),   # R7
    ctrl.Rule(temperatura['Confort']     & tasa_cambio['Estable'],  potencia['Neutro']),          # R8
    ctrl.Rule(temperatura['Confort']     & tasa_cambio['Subiendo'], potencia['EnfriarPoco']),    # R9
    # Temperatura Caliente
    ctrl.Rule(temperatura['Caliente']    & tasa_cambio['Bajando'],  potencia['Neutro']),          # R10
    ctrl.Rule(temperatura['Caliente']    & tasa_cambio['Estable'],  potencia['EnfriarPoco']),    # R11
    ctrl.Rule(temperatura['Caliente']    & tasa_cambio['Subiendo'], potencia['EnfriarMucho']),   # R12
    # Temperatura MuyCaliente
    ctrl.Rule(temperatura['MuyCaliente'] & tasa_cambio['Bajando'],  potencia['EnfriarPoco']),    # R13
    ctrl.Rule(temperatura['MuyCaliente'] & tasa_cambio['Estable'],  potencia['EnfriarMucho']),   # R14
    ctrl.Rule(temperatura['MuyCaliente'] & tasa_cambio['Subiendo'], potencia['EnfriarMucho']),   # R15
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. SISTEMA DE CONTROL - MÉTODO MAMDANI (Centroide)
# ─────────────────────────────────────────────────────────────────────────────
sistema_ctrl   = ctrl.ControlSystem(reglas)
simulador      = ctrl.ControlSystemSimulation(sistema_ctrl)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNCIÓN DE DEFUZZIFICACIÓN MANUAL (Bisector y Centroide)
# ─────────────────────────────────────────────────────────────────────────────
def trapz(y, x):
    """Compatibilidad numpy 1.x / 2.x."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

def defuzzificar_bisector(x, mf):
    """Método del Bisector: divide el área en dos partes iguales."""
    area_total = trapz(mf, x)
    if area_total == 0:
        return 0.0
    area_acum = 0.0
    for i in range(1, len(x)):
        area_acum += trapz(mf[i-1:i+1], x[i-1:i+1])
        if area_acum >= area_total / 2:
            return x[i]
    return x[-1]

def defuzzificar_centroide(x, mf):
    """Método del Centroide: centro de gravedad del área."""
    area = trapz(mf, x)
    if area == 0:
        return 0.0
    return trapz(mf * x, x) / area

def obtener_mf_agregada(t_val, dt_val):
    """Calcula la función de membresía agregada de la salida para un escenario."""
    # Fuzzificación de entradas
    mf_temp = {
        'MuyFria':     fuzz.trapmf(temperatura_range, [10, 10, 13, 16]),
        'Fria':        fuzz.trimf(temperatura_range, [13, 16, 19]),
        'Confort':     fuzz.gaussmf(temperatura_range, 20, 1),
        'Caliente':    fuzz.trimf(temperatura_range, [19, 22, 25]),
        'MuyCaliente': fuzz.trapmf(temperatura_range, [24, 27, 30, 30]),
    }
    mf_dt = {
        'Bajando':  fuzz.trapmf(tasa_cambio_range, [-3, -3, -1, 0]),
        'Estable':  fuzz.gaussmf(tasa_cambio_range, 0, 0.5),
        'Subiendo': fuzz.trapmf(tasa_cambio_range, [0, 1, 3, 3]),
    }
    mf_sal = {
        'EnfriarMucho':  fuzz.trapmf(potencia_range, [-100, -100, -70, -40]),
        'EnfriarPoco':   fuzz.trimf(potencia_range, [-60, -30, 0]),
        'Neutro':        fuzz.gaussmf(potencia_range, 0, 10),
        'CalentarPoco':  fuzz.trimf(potencia_range, [0, 30, 60]),
        'CalentarMucho': fuzz.trapmf(potencia_range, [40, 70, 100, 100]),
    }

    # Grados de membresía para las entradas
    mu_temp = {k: fuzz.interp_membership(temperatura_range, v, t_val)
                for k, v in mf_temp.items()}
    mu_dt   = {k: fuzz.interp_membership(tasa_cambio_range, v, dt_val)
                for k, v in mf_dt.items()}

    # Definición de reglas (antecedente_temp, antecedente_dt, consecuente)
    reglas_manuales = [
        ('MuyFria',     'Bajando',  'CalentarMucho'),
        ('MuyFria',     'Estable',  'CalentarMucho'),
        ('MuyFria',     'Subiendo', 'CalentarPoco'),
        ('Fria',        'Bajando',  'CalentarMucho'),
        ('Fria',        'Estable',  'CalentarPoco'),
        ('Fria',        'Subiendo', 'Neutro'),
        ('Confort',     'Bajando',  'CalentarPoco'),
        ('Confort',     'Estable',  'Neutro'),
        ('Confort',     'Subiendo', 'EnfriarPoco'),
        ('Caliente',    'Bajando',  'Neutro'),
        ('Caliente',    'Estable',  'EnfriarPoco'),
        ('Caliente',    'Subiendo', 'EnfriarMucho'),
        ('MuyCaliente', 'Bajando',  'EnfriarPoco'),
        ('MuyCaliente', 'Estable',  'EnfriarMucho'),
        ('MuyCaliente', 'Subiendo', 'EnfriarMucho'),
    ]

    # Agregación: máximo de las salidas recortadas (Mamdani)
    mf_agregada = np.zeros_like(potencia_range)
    reglas_activadas = []

    for t_ant, dt_ant, consecuente in reglas_manuales:
        fuerza = min(mu_temp[t_ant], mu_dt[dt_ant])
        if fuerza > 0.001:
            recortado = np.fmin(fuerza, mf_sal[consecuente])
            mf_agregada = np.fmax(mf_agregada, recortado)
            reglas_activadas.append((t_ant, dt_ant, consecuente, round(fuerza, 4)))

    return mf_agregada, reglas_activadas

# ─────────────────────────────────────────────────────────────────────────────
# 7. GRÁFICAS DE FUNCIONES DE MEMBRESÍA
# ─────────────────────────────────────────────────────────────────────────────
colores = ['#E74C3C', '#E67E22', '#2ECC71', '#3498DB', '#9B59B6']

def graficar_variable(ax, x_range, variable_ctrl, titulo, xlabel, colores_):
    for (nombre, mf_vals), color in zip(variable_ctrl.terms.items(), colores_):
        ax.plot(x_range, mf_vals.mf, color=color, linewidth=2.5, label=nombre)
        ax.fill_between(x_range, mf_vals.mf, alpha=0.12, color=color)
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Grado de membresía μ', fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Funciones de Membresía — Sistema de Control Difuso Térmico',
            fontsize=14, fontweight='bold', y=1.02)

graficar_variable(axes[0], temperatura_range, temperatura,
                'Variable: Temperatura (Entrada)', 'Temperatura (°C)', colores)
graficar_variable(axes[1], tasa_cambio_range, tasa_cambio,
                'Variable: Tasa de Cambio ΔT (Entrada)', 'Tasa de Cambio (°C/min)',
                ['#3498DB', '#2ECC71', '#E74C3C'])
graficar_variable(axes[2], potencia_range, potencia,
                'Variable: Potencia de Control (Salida)', 'Potencia (%)', colores)

# Línea de zona confort en temperatura
axes[0].axvspan(18, 22, alpha=0.08, color='green', label='Zona confort')
axes[0].axvline(18, color='green', linestyle=':', alpha=0.6, linewidth=1)
axes[0].axvline(22, color='green', linestyle=':', alpha=0.6, linewidth=1)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. ESCENARIOS DE PRUEBA
# ─────────────────────────────────────────────────────────────────────────────
escenarios = [
    {"id": 1, "T": 16.0, "dT": 0.0,  "desc": "Frío estable",          "esperado": "Calentar moderado"},
    {"id": 2, "T": 20.0, "dT": 0.0,  "desc": "Confort estable",        "esperado": "Neutro (sin acción)"},
    {"id": 3, "T": 25.0, "dT": 1.0,  "desc": "Caliente subiendo",      "esperado": "Enfriar mucho"},
    {"id": 4, "T": 14.0, "dT": -1.0, "desc": "Muy frío bajando",       "esperado": "Calentar mucho"},
    {"id": 5, "T": 22.0, "dT": 2.0,  "desc": "Confort pero subiendo",  "esperado": "Enfriar poco"},
]

resultados = []

print("\n" + "="*70)
print("  SIMULACIÓN DE ESCENARIOS — MÉTODO MAMDANI")
print("="*70)

for esc in escenarios:
    T_val  = esc["T"]
    dT_val = esc["dT"]

    # --- Centroide vía skfuzzy ---
    simulador.input['temperatura'] = T_val
    simulador.input['tasa_cambio'] = dT_val
    simulador.compute()
    centroide_val = simulador.output['potencia']

    # --- Defuzzificación manual (Centroide y Bisector) ---
    mf_ag, reglas_act = obtener_mf_agregada(T_val, dT_val)
    centroide_manual  = defuzzificar_centroide(potencia_range, mf_ag)
    bisector_val      = defuzzificar_bisector(potencia_range, mf_ag)

    esc['centroide']         = round(centroide_val, 2)
    esc['centroide_manual']  = round(centroide_manual, 2)
    esc['bisector']          = round(bisector_val, 2)
    esc['mf_agregada']       = mf_ag
    esc['reglas_activadas']  = reglas_act

    resultados.append(esc)

    print(f"\n Escenario {esc['id']}: {esc['desc']}")
    print(f"   Entrada:  T={T_val}°C,  ΔT={dT_val} °C/min")
    print(f"   Esperado: {esc['esperado']}")
    print(f"   ─── Reglas activadas:")
    for r in reglas_act:
        print(f"       SI Temp={r[0]} Y ΔT={r[1]} → {r[2]}   (fuerza={r[3]})")
    print(f"   ─── Defuzzificación:")
    print(f"       Centroide (skfuzzy):  {esc['centroide']:+.2f}%")
    print(f"       Centroide (manual):   {esc['centroide_manual']:+.2f}%")
    print(f"       Bisector  (manual):   {esc['bisector']:+.2f}%")

print("\n" + "="*70)

# ─────────────────────────────────────────────────────────────────────────────
# 9. GRÁFICA COMPARATIVA CENTROIDE vs BISECTOR
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Defuzzificación por Escenario — Centroide vs Bisector',
                fontsize=14, fontweight='bold')

esc_labels = [f"E{e['id']}\n{e['desc']}" for e in resultados]
cog_vals   = [e['centroide_manual'] for e in resultados]
bis_vals   = [e['bisector']         for e in resultados]

# ── Subplots individuales por escenario ──
for idx, esc in enumerate(resultados):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    mf = esc['mf_agregada']

    ax.fill_between(potencia_range, mf, alpha=0.35, color='#3498DB', label='Área agregada')
    ax.plot(potencia_range, mf, color='#2980B9', linewidth=1.8)

    cog = esc['centroide_manual']
    bis = esc['bisector']
    ax.axvline(cog, color='#E74C3C', linewidth=2.2, linestyle='-',
                label=f'Centroide: {cog:+.1f}%')
    ax.axvline(bis, color='#27AE60', linewidth=2.2, linestyle='--',
                label=f'Bisector: {bis:+.1f}%')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    ax.set_title(f"Esc. {esc['id']}: {esc['desc']}\nT={esc['T']}°C, ΔT={esc['dT']}°C/min",
                    fontsize=10, fontweight='bold')
    ax.set_xlabel('Potencia (%)', fontsize=9)
    ax.set_ylabel('μ', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_xlim(-105, 105)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Gráfica comparativa global ──
ax_bar = axes[1][2]
x_pos  = np.arange(len(resultados))
width  = 0.35
bars1  = ax_bar.bar(x_pos - width/2, cog_vals, width, color='#E74C3C',
                    alpha=0.85, label='Centroide', edgecolor='white')
bars2  = ax_bar.bar(x_pos + width/2, bis_vals,   width, color='#27AE60',
                    alpha=0.85, label='Bisector',  edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., h + (2 if h >= 0 else -5),
                f'{h:.0f}', ha='center', va='bottom', fontsize=8, color='#E74C3C')
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., h + (2 if h >= 0 else -5),
                f'{h:.0f}', ha='center', va='bottom', fontsize=8, color='#27AE60')

ax_bar.axhline(0, color='black', linewidth=0.8)
ax_bar.axhspan(-100, -0.1, alpha=0.04, color='blue')
ax_bar.axhspan(0.1, 100,   alpha=0.04, color='red')
ax_bar.set_title('Comparación Global\nCentroide vs Bisector', fontsize=10, fontweight='bold')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([f'E{e["id"]}' for e in resultados], fontsize=9)
ax_bar.set_ylabel('Potencia de salida (%)', fontsize=9)
ax_bar.legend(fontsize=9)
ax_bar.set_ylim(-110, 110)
ax_bar.grid(True, alpha=0.25, axis='y', linestyle='--')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 10. GRÁFICA DETALLADA ESCENARIO REQUERIDO: T=16°C
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Análisis Detallado — Escenario Requerido: T=16°C, ΔT=0 °C/min',
            fontsize=13, fontweight='bold')

# Subplot 1: Fuzzificación de entradas
ax1 = axes[0]
T_req  = 16.0
dT_req = 0.0

# Temperatura
mf_T = {
    'MuyFria': fuzz.trapmf(temperatura_range, [10, 10, 13, 16]),
    'Fria':    fuzz.trimf(temperatura_range, [13, 16, 19]),
    'Confort': fuzz.gaussmf(temperatura_range, 20, 1),
}
colores_t = ['#E74C3C', '#E67E22', '#2ECC71']
for (nombre, vals), color in zip(mf_T.items(), colores_t):
    ax1.plot(temperatura_range, vals, color=color, linewidth=2.5, label=nombre)
    ax1.fill_between(temperatura_range, vals, alpha=0.12, color=color)

mu_fria    = fuzz.interp_membership(temperatura_range, fuzz.trimf(temperatura_range, [13,16,19]), T_req)
mu_muyfria = fuzz.interp_membership(temperatura_range, fuzz.trapmf(temperatura_range,[10,10,13,16]), T_req)

ax1.axvline(T_req, color='black', linewidth=2, linestyle='--', label=f'T = {T_req}°C')
ax1.plot([T_req], [mu_fria],    'o', color='#E67E22', markersize=10,
        label=f'μ_Fria = {mu_fria:.2f}',    zorder=5)
ax1.plot([T_req], [mu_muyfria], 's', color='#E74C3C', markersize=10,
        label=f'μ_MuyFria = {mu_muyfria:.2f}', zorder=5)

ax1.annotate(f'μ={mu_fria:.2f}',    xy=(T_req, mu_fria),    xytext=(17.5, mu_fria+0.05),
            arrowprops=dict(arrowstyle='->', color='#E67E22'), fontsize=9, color='#E67E22')
ax1.annotate(f'μ={mu_muyfria:.2f}', xy=(T_req, mu_muyfria), xytext=(17.5, mu_muyfria+0.1),
            arrowprops=dict(arrowstyle='->', color='#E74C3C'), fontsize=9, color='#E74C3C')

ax1.set_title('Fuzzificación: T = 16°C', fontsize=11, fontweight='bold')
ax1.set_xlabel('Temperatura (°C)', fontsize=10)
ax1.set_ylabel('Grado de membresía μ', fontsize=10)
ax1.set_xlim(10, 25)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot 2: MF agregada con líneas de defuzz
ax2 = axes[1]
mf_ag_req, reg_act_req = obtener_mf_agregada(T_req, dT_req)
cog_req = defuzzificar_centroide(potencia_range, mf_ag_req)
bis_req = defuzzificar_bisector(potencia_range, mf_ag_req)

ax2.fill_between(potencia_range, mf_ag_req, alpha=0.4, color='#3498DB', label='Área agregada (Mamdani)')
ax2.plot(potencia_range, mf_ag_req, color='#2980B9', linewidth=2)
ax2.axvline(cog_req, color='#E74C3C', linewidth=2.5, linestyle='-',
            label=f'Centroide = {cog_req:+.1f}%')
ax2.axvline(bis_req, color='#27AE60', linewidth=2.5, linestyle='--',
            label=f'Bisector = {bis_req:+.1f}%')
ax2.axvline(0, color='gray', linewidth=1, linestyle=':', alpha=0.5)

# Marcar área del centroide
idx_cog = np.argmin(np.abs(potencia_range - cog_req))
ax2.fill_between(potencia_range[:idx_cog], mf_ag_req[:idx_cog],
                alpha=0.15, color='#E74C3C')

ax2.set_title('Defuzzificación: Resultado para T=16°C, ΔT=0', fontsize=11, fontweight='bold')
ax2.set_xlabel('Potencia de Control (%)', fontsize=10)
ax2.set_ylabel('μ agregado', fontsize=10)
ax2.legend(fontsize=10)
ax2.set_xlim(-10, 110)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 11. RETORNO DE DATOS PARA EL INFORME
# ─────────────────────────────────────────────────────────────────────────────
print("\n Simulación completa. Las 3 gráficas se han mostrado en pantalla.")
print("\n─── TABLA RESUMEN ───")
print(f"{'Esc':>4} {'Descripción':<25} {'T(°C)':>6} {'ΔT':>5} {'Centroide':>10} {'Bisector':>9} {'Diferencia':>11}")
print("─" * 75)
for e in resultados:
    dif = abs(e['centroide_manual'] - e['bisector'])
    print(f"{e['id']:>4} {e['desc']:<25} {e['T']:>6.1f} {e['dT']:>+5.1f} "
        f"{e['centroide_manual']:>+10.2f} {e['bisector']:>+9.2f} {dif:>11.2f}")
print("─" * 75)