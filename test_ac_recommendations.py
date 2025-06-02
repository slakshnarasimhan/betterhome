import json
from combined_script import generate_final_product_list, get_specific_product_recommendations

def test_ac_recommendations():
    # Sample user data with AC requirements
    user_data = {
        'total_budget': 500000,
        'demographics': {
            'adults': 2,
            'children': 1,
            'seniors': 1
        },
        'hall': {
            'ac': 'yes',
            'fans': 2,
            'color_theme': 'white'
        },
        'master_bedroom': {
            'ac': 'yes',
            'fans': 1,
            'color_theme': 'white',
            'bathroom': {
                'water_heater_type': 'storage',
                'exhaust_fan_size': '6 inch'
            }
        },
        'bedroom_2': {
            'ac': 'yes',
            'fans': 1,
            'color_theme': 'white',
            'bathroom': {
                'water_heater_type': 'storage',
                'exhaust_fan_size': '6 inch'
            }
        },
        'kitchen': {
            'chimney_width': '90cm',
            'gas_stove_type': 'glass top',
            'gas_stove_burners': '4',
            'small_fan': 'yes'
        },
        'laundry': {
            'washing_machine': 'front load',
            'dryer': 'yes'
        }
    }

    # Generate final product list
    final_list = generate_final_product_list(user_data)

    # Print AC recommendations for each room
    print("\nAC Recommendations Test Results:")
    print("===============================")
    
    rooms = ['hall', 'master_bedroom', 'bedroom_2']
    for room in rooms:
        print(f"\n{room.replace('_', ' ').title()}:")
        if final_list[room]['ac']:
            for ac in final_list[room]['ac']:
                print(f"- Brand: {ac['brand']}")
                print(f"  Model: {ac['model']}")
                print(f"  Price: â‚¹{ac['price']:,.2f}")
                print(f"  Features: {', '.join(ac['features'])}")
                print(f"  Color Options: {', '.join(ac['color_options'])}")
                print(f"  Color Match: {ac['color_match']}")
        else:
            print("No AC recommendations found")

if __name__ == "__main__":
    test_ac_recommendations() 