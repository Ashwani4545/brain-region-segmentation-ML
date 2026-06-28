import os

class TelehealthRouter:
    def __init__(self):
        # Simulated Doctor Directory
        self.doctors = [
            {
                'name': "Dr. Aravind Sharma",
                'specialty': "Neurology",
                'hospital': "Apollo Brain Spine Center",
                'city': "Bangalore",
                'lat': 12.9716, 'lon': 77.5946,
                'rating': 4.9
            },
            {
                'name': "Dr. Priya Patel",
                'specialty': "Neurology",
                'hospital': "Max Super Speciality Hospital",
                'city': "Delhi",
                'lat': 28.7041, 'lon': 77.1025,
                'rating': 4.8
            },
            {
                'name': "Dr. Rajesh Verma",
                'specialty': "Pulmonology",
                'hospital': "Fortis Chest Clinic",
                'city': "Mumbai",
                'lat': 19.0760, 'lon': 72.8777,
                'rating': 4.7
            },
            {
                'name': "Dr. Sanjay Mehta",
                'specialty': "Pulmonology",
                'hospital': "Narayana Pulmonary Care",
                'city': "Bangalore",
                'lat': 12.9800, 'lon': 77.6100,
                'rating': 4.9
            },
            {
                'name': "Dr. Vikram Sen",
                'specialty': "Cardiology",
                'hospital': "Medanta Heart Institute",
                'city': "Delhi",
                'lat': 28.6100, 'lon': 77.2000,
                'rating': 4.9
            },
            {
                'name': "Dr. Ananya Reddy",
                'specialty': "Cardiology",
                'hospital': "Apollo Heart Centre",
                'city': "Chennai",
                'lat': 13.0827, 'lon': 80.2707,
                'rating': 4.8
            },
            {
                'name': "Dr. Neha Rao",
                'specialty': "General Medicine",
                'hospital': "Manipal Diagnostics Clinic",
                'city': "Bangalore",
                'lat': 12.9300, 'lon': 77.5700,
                'rating': 4.6
            },
            {
                'name': "Dr. Sunil Gupta",
                'specialty': "General Medicine",
                'hospital': "Lilavati Family Clinic",
                'city': "Mumbai",
                'lat': 19.0500, 'lon': 72.8300,
                'rating': 4.7
            }
        ]
        
        # Simple coordinate map for city fallback distance estimations
        self.city_coords = {
            'bangalore': (12.9716, 77.5946),
            'delhi': (28.7041, 77.1025),
            'mumbai': (19.0760, 72.8777),
            'chennai': (13.0827, 80.2707)
        }

    def get_closest_specialists(self, modality: str, patient_city: str) -> list:
        """
        Query the doctor registry, filter by clinical specialty (matching scan modality),
        and sort specialists by distance based on patient location.
        """
        # Map scan modality to specialist type
        target_specialty = "General Medicine"
        if modality == 'CT':
            target_specialty = "Neurology"
        elif modality == 'CXR':
            target_specialty = "Pulmonology"
        elif modality == 'ECG':
            target_specialty = "Cardiology"
            
        p_city = str(patient_city).strip().lower()
        p_lat, p_lon = self.city_coords.get(p_city, (12.9716, 77.5946)) # default Bangalore
        
        results = []
        for doc in self.doctors:
            if doc['specialty'] == target_specialty:
                # Estimate distance using Haversine approximation
                dist = self._haversine(p_lat, p_lon, doc['lat'], doc['lon'])
                
                # Copy doctor and inject distance
                doc_entry = dict(doc)
                doc_entry['distance_km'] = round(dist, 1)
                results.append(doc_entry)
                
        # Sort by distance ascending
        results.sort(key=lambda x: x['distance_km'])
        return results

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Estimate straight-line distance in kilometers."""
        import math
        # Radius of Earth in km
        R = 6371.0
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

_router = None

def get_telehealth_router() -> TelehealthRouter:
    global _router
    if _router is None:
        _router = TelehealthRouter()
    return _router
