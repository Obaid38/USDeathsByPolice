import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class PoliceShootingsAnalysis:
    def __init__(self):
        self.shootings_data = None
        self.census_data = None
        self.merged_data = None

    def load_data(self):
        try:
            print("Loading Washington Post police shootings data...")
            self.shootings_data = pd.read_csv(
                r'C:\python projects pycharm\DataScienceDeathsUSPoliceAnalysis\fatal-police-shootings-data.csv'
            )

            self.shootings_data['date'] = pd.to_datetime(self.shootings_data['date'])
            self.shootings_data['year'] = self.shootings_data['date'].dt.year

            print(
                f"Loaded {len(self.shootings_data)} police shooting records from {self.shootings_data['year'].min()} to {self.shootings_data['year'].max()}")

            print("Note: For demonstration purposes, using sample census data structure.")


            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def explore_basic_statistics(self):
        if self.shootings_data is None:
            print("Please load data first.")
            return

        print("=== BASIC DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.shootings_data.shape}")
        print(f"Date range: {self.shootings_data['date'].min()} to {self.shootings_data['date'].max()}")
        print(f"\nColumns: {list(self.shootings_data.columns)}")

        print("\n=== YEARLY TRENDS ===")
        yearly_counts = self.shootings_data.groupby('year').size()
        print(yearly_counts)

        print("\n=== RACE/ETHNICITY BREAKDOWN ===")
        if 'race' in self.shootings_data.columns:
            race_counts = self.shootings_data['race'].value_counts()
            race_percentages = (race_counts / len(self.shootings_data) * 100).round(2)
            for race, count in race_counts.items():
                print(f"{race}: {count} ({race_percentages[race]}%)")

        print("\n=== AGE STATISTICS ===")
        if 'age' in self.shootings_data.columns:
            age_stats = self.shootings_data['age'].describe()
            print(age_stats)

        print("\n=== GEOGRAPHIC DISTRIBUTION (Top 10 States) ===")
        if 'state' in self.shootings_data.columns:
            state_counts = self.shootings_data['state'].value_counts().head(10)
            print(state_counts)

    def analyze_temporal_trends(self):
        if self.shootings_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        yearly_counts = self.shootings_data.groupby('year').size()
        axes[0, 0].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Fatal Police Shootings by Year', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Fatal Shootings')
        axes[0, 0].grid(True, alpha=0.3)

        self.shootings_data['month'] = self.shootings_data['date'].dt.month
        monthly_avg = self.shootings_data.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0, 1].bar(range(1, 13), monthly_avg.values, color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Average Fatal Shootings by Month', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Number of Shootings')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names, rotation=45)

        self.shootings_data['day_of_week'] = self.shootings_data['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.shootings_data['day_of_week'].value_counts()[day_order]
        axes[1, 0].bar(range(7), day_counts.values, color='coral', alpha=0.7)
        axes[1, 0].set_title('Fatal Shootings by Day of Week', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Total Number of Shootings')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels([day[:3] for day in day_order], rotation=45)

        if 'age' in self.shootings_data.columns:
            age_clean = self.shootings_data['age'].dropna()
            axes[1, 1].hist(age_clean, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(age_clean.mean(), color='red', linestyle='--',
                               label=f'Mean: {age_clean.mean():.1f}')
            axes[1, 1].axvline(age_clean.median(), color='blue', linestyle='--',
                               label=f'Median: {age_clean.median():.1f}')
            axes[1, 1].set_title('Age Distribution of Victims', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Age')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def analyze_demographic_patterns(self):
        if self.shootings_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        if 'race' in self.shootings_data.columns:
            race_counts = self.shootings_data['race'].value_counts()

            axes[0, 0].pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%',
                           startangle=90)
            axes[0, 0].set_title('Fatal Shootings by Race/Ethnicity', fontsize=14, fontweight='bold')

            us_population_race = {
                'W': 72.4,
                'B': 12.6,
                'H': 16.3,
                'A': 4.8,
                'N': 0.9,
                'O': 2.9
            }

            shooting_percentages = (race_counts / race_counts.sum() * 100)
            comparison_data = []

            for race in race_counts.index:
                if race in us_population_race:
                    shooting_pct = shooting_percentages[race]
                    pop_pct = us_population_race[race]
                    comparison_data.append({
                        'Race': race,
                        'Shooting_Rate': shooting_pct,
                        'Population_Rate': pop_pct,
                        'Ratio': shooting_pct / pop_pct
                    })

            comparison_df = pd.DataFrame(comparison_data)

            x = np.arange(len(comparison_df))
            width = 0.35

            axes[0, 1].bar(x - width / 2, comparison_df['Shooting_Rate'], width,
                           label='% of Shootings', alpha=0.8)
            axes[0, 1].bar(x + width / 2, comparison_df['Population_Rate'], width,
                           label='% of US Population', alpha=0.8)
            axes[0, 1].set_xlabel('Race/Ethnicity')
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].set_title('Shootings vs Population Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(comparison_df['Race'])
            axes[0, 1].legend()

        if 'gender' in self.shootings_data.columns:
            gender_counts = self.shootings_data['gender'].value_counts()
            axes[1, 0].bar(gender_counts.index, gender_counts.values,
                           color=['lightblue', 'pink'], alpha=0.7)
            axes[1, 0].set_title('Fatal Shootings by Gender', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Gender')
            axes[1, 0].set_ylabel('Number of Shootings')

            total = gender_counts.sum()
            for i, v in enumerate(gender_counts.values):
                axes[1, 0].text(i, v + total * 0.01, f'{v}\n({v / total * 100:.1f}%)',
                                ha='center', fontweight='bold')

        if 'armed' in self.shootings_data.columns:
            armed_simplified = self.shootings_data['armed'].fillna('unknown')
            armed_simplified = armed_simplified.replace({
                'gun': 'firearm',
                'knife': 'knife/blade',
                'unarmed': 'unarmed',
                'unknown weapon': 'unknown'
            })

            armed_counts = armed_simplified.value_counts()
            top_categories = armed_counts.head(8)
            other_count = armed_counts.iloc[8:].sum()
            if other_count > 0:
                top_categories['other'] = other_count

            axes[1, 1].barh(range(len(top_categories)), top_categories.values, alpha=0.7)
            axes[1, 1].set_yticks(range(len(top_categories)))
            axes[1, 1].set_yticklabels(top_categories.index)
            axes[1, 1].set_title('Fatal Shootings by Armed Status', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Number of Shootings')

        plt.tight_layout()
        plt.show()

    def analyze_geographic_patterns(self):
        if self.shootings_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        if 'state' in self.shootings_data.columns:
            state_counts = self.shootings_data['state'].value_counts().head(15)

            axes[0, 0].barh(range(len(state_counts)), state_counts.values, alpha=0.7)
            axes[0, 0].set_yticks(range(len(state_counts)))
            axes[0, 0].set_yticklabels(state_counts.index)
            axes[0, 0].set_title('Top 15 States by Fatal Shootings', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Number of Shootings')

            axes[0, 1].text(0.5, 0.5, 'Per Capita Analysis\n(Requires State Population Data)',
                            ha='center', va='center', transform=axes[0, 1].transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[0, 1].set_title('Per Capita Rates by State', fontsize=14, fontweight='bold')

        if 'city' in self.shootings_data.columns:
            city_counts = self.shootings_data['city'].value_counts().head(15)

            axes[1, 0].barh(range(len(city_counts)), city_counts.values, alpha=0.7, color='coral')
            axes[1, 0].set_yticks(range(len(city_counts)))
            axes[1, 0].set_yticklabels(city_counts.index)
            axes[1, 0].set_title('Top 15 Cities by Fatal Shootings', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Number of Shootings')

        region_mapping = {
            'Northeast': ['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'],
            'Southeast': ['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV'],
            'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
            'Southwest': ['AZ', 'NM', 'OK', 'TX'],
            'West': ['AK', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'OR', 'UT', 'WA', 'WY'],
            'Other': ['DC', 'DE', 'MD']
        }

        if 'state' in self.shootings_data.columns:
            state_to_region = {}
            for region, states in region_mapping.items():
                for state in states:
                    state_to_region[state] = region

            self.shootings_data['region'] = self.shootings_data['state'].map(state_to_region)
            region_counts = self.shootings_data['region'].value_counts()

            axes[1, 1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%',
                           startangle=90)
            axes[1, 1].set_title('Fatal Shootings by US Region', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def generate_summary_report(self):
        if self.shootings_data is None:
            print("Please load data first.")
            return

        print("=" * 60)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)

        total_shootings = len(self.shootings_data)
        years_covered = self.shootings_data['year'].nunique()
        avg_per_year = total_shootings / years_covered

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total fatal shootings: {total_shootings:,}")
        print(f"   • Years covered: {years_covered}")
        print(f"   • Average per year: {avg_per_year:.1f}")
        print(
            f"   • Date range: {self.shootings_data['date'].min().strftime('%Y-%m-%d')} to {self.shootings_data['date'].max().strftime('%Y-%m-%d')}")

        if 'race' in self.shootings_data.columns:
            race_counts = self.shootings_data['race'].value_counts()
            print(f"\n👥 DEMOGRAPHIC BREAKDOWN:")
            for race, count in race_counts.head(5).items():
                pct = count / total_shootings * 100
                print(f"   • {race}: {count:,} ({pct:.1f}%)")

        if 'age' in self.shootings_data.columns:
            age_stats = self.shootings_data['age'].describe()
            print(f"\n📈 AGE STATISTICS:")
            print(f"   • Average age: {age_stats['mean']:.1f} years")
            print(f"   • Median age: {age_stats['50%']:.1f} years")
            print(f"   • Age range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")

        if 'state' in self.shootings_data.columns:
            top_states = self.shootings_data['state'].value_counts().head(5)
            print(f"\n🗺️  TOP 5 STATES:")
            for state, count in top_states.items():
                pct = count / total_shootings * 100
                print(f"   • {state}: {count:,} ({pct:.1f}%)")

        yearly_trend = self.shootings_data.groupby('year').size()
        trend_change = ((yearly_trend.iloc[-1] - yearly_trend.iloc[0]) / yearly_trend.iloc[0]) * 100

        print(f"\n📅 TEMPORAL PATTERNS:")
        print(f"   • Highest year: {yearly_trend.idxmax()} ({yearly_trend.max()} shootings)")
        print(f"   • Lowest year: {yearly_trend.idxmin()} ({yearly_trend.min()} shootings)")
        print(f"   • Overall trend: {trend_change:+.1f}% change from first to last year")

        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"   • This analysis is based on reported data and may not capture all incidents")
        print(f"   • Statistical patterns do not imply causation")
        print(f"   • Further analysis with socioeconomic data would provide deeper insights")
        print(f"   • Data quality and reporting consistency may vary across jurisdictions")

        print("=" * 60)


def main():
    print("Police Fatal Shootings Analysis")
    print("=" * 50)

    analyzer = PoliceShootingsAnalysis()

    if not analyzer.load_data():
        print("Failed to load data. Please check your internet connection.")
        return

    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    analyzer.explore_basic_statistics()

    print("\n" + "=" * 50)
    print("TEMPORAL ANALYSIS")
    print("=" * 50)
    analyzer.analyze_temporal_trends()

    print("\n" + "=" * 50)
    print("DEMOGRAPHIC ANALYSIS")
    print("=" * 50)
    analyzer.analyze_demographic_patterns()

    print("\n" + "=" * 50)
    print("GEOGRAPHIC ANALYSIS")
    print("=" * 50)
    analyzer.analyze_geographic_patterns()

    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    analyzer.generate_summary_report()


if __name__ == "__main__":
    main()


def correlation_analysis(df, socioeconomic_data=None):
    if socioeconomic_data is None:
        print("Socioeconomic data needed for correlation analysis")
        return

    pass


def predictive_modeling(df):
    print("Predictive modeling would require:")
    print("1. Feature engineering from available data")
    print("2. Time series analysis for trend prediction")
    print("3. Geographic clustering analysis")
    print("4. Risk factor identification")

    pass


def policy_impact_analysis(df, policy_dates=None):
    print("Policy impact analysis would examine:")
    print("1. Before/after comparisons for policy implementations")
    print("2. Difference-in-differences analysis across jurisdictions")
    print("3. Interrupted time series analysis")

    pass