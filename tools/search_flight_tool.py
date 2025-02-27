import os
import requests
import pandas as pd
from dotenv import load_dotenv
from smolagents import Tool
import re


class AmadeusFlightSearchTool(Tool):
    name = "amadeus_flight_search"
    description = "Finds direct flights between two airports and looks for the best offers. Should be used as default tool for finding flights."
    inputs = {
        "departure_city": {
            "type": "string",
            "description": "Name of the departure city. Example 'New York'.",
        },
        "departure_country": {
            "type": "string",
            "description": "Country of the departure city. Example 'United States of America'.",
        },
        "destination_city": {
            "type": "string",
            "description": "Name of the destination city in natural language. Example 'London'.",
        },
        "destination_country": {
            "type": "string",
            "description": "Country of the destination city. Example 'United Kingdom'.",
        },
        "travel_date": {
            "type": "string",
            "description": "Travel date in YYYY-MM-DD format.",
        },
        "currency": {
            "type": "string",
            "description": "Optional currency code. Example: If the user wants to use euro currency, the code would be 'EUR'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_key = os.getenv("AMADEUS_API_KEY")
        self.api_secret = os.getenv("AMADEUS_API_SECRET")
        self.base_url = "https://test.api.amadeus.com"

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Missing Amadeus API credentials. Please make sure the environment variables are set."
            )

        self.token = self.get_access_token()

    def get_access_token(self):
        """Fetches the OAuth token from Amadeus."""
        url = f"{self.base_url}/v1/security/oauth2/token"
        response = requests.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret,
            },
        )
        response.raise_for_status()
        return response.json().get("access_token")

    def fetch_flights(
        self,
        departure_airport: str,
        destination_airport: str,
        travel_date: str,
        currency: str = None,
    ):
        """Fetches direct flights based on input criteria."""
        url = f"{self.base_url}/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {
            "originLocationCode": departure_airport,
            "destinationLocationCode": destination_airport,
            "departureDate": travel_date,
            "adults": 1,
            "nonStop": "true",
            "currencyCode": currency if currency else "USD",
            "max": 10,
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 400:
            raise ValueError(
                "Error fetching flights: Bad request. Check your input parameters."
            )
        response.raise_for_status()
        return response.json().get("data", [])

    def get_airline_name(self, carrier_code: str):
        """Fetches the full airline name given an airline IATA code."""
        url = f"{self.base_url}/v1/reference-data/airlines"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {"airlineCodes": carrier_code}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200 and response.json().get("data"):
            return response.json()["data"][0].get("businessName", carrier_code)
        return carrier_code

    def get_airport_code(self, city_name: str, country_code: str) -> str:
        """Retrieves the IATA airport code for a given city."""
        url = f"{self.base_url}/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {"keyword": city_name, "subType": "AIRPORT"}

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        airports = response.json().get("data", [])
        if country_code:
            airports = [
                airport
                for airport in airports
                if airport.get("address", {}).get("countryCode") == country_code
            ]

        if not airports:
            raise ValueError(
                f"Could not find an airport in {city_name}, {country_code}. Country codes could be wrong or country is not available."
            )

        return airports[0]["iataCode"]

    def convert_country_to_code(self, country_name: str) -> str:
        """Converts a full country name to an ISO Alpha-2 country code (e.g., 'United Kingdom' -> 'GB')."""
        url = f"https://restcountries.com/v3.1/name/{country_name}?fields=cca2"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(
                f"Could not convert country name: {country_name}. Check spelling."
            )

        country_data = response.json()
        return country_data[0]["cca2"]

    def forward(
        self,
        departure_city: str,
        departure_country: str,
        destination_city: str,
        destination_country: str,
        travel_date: str,
        currency: str = None,
    ) -> str:
        """Main function that returns flight data as a formatted string."""
        # Validate date format...
        if not re.match(r"\d{4}-\d{2}-\d{2}", travel_date):
            raise ValueError(
                "Invalid date format. Please use YYYY-MM-DD format for the travel date."
            )

        # Get country codes for departure and destination
        departure_country_code = self.convert_country_to_code(departure_country)
        destination_country_code = self.convert_country_to_code(destination_country)

        # Get airport codes for departure and destination
        departure_airport = self.get_airport_code(
            departure_city, departure_country_code
        )
        destination_airport = self.get_airport_code(
            destination_city, destination_country_code
        )

        currency = currency if currency else "USD"

        # Look for direct flights
        flights = self.fetch_flights(
            departure_airport, destination_airport, travel_date, currency
        )
        flight_list = []

        for flight in flights:
            itinerary = flight.get("itineraries", [{}])[0].get("segments", [{}])[0]
            airline_name = self.get_airline_name(
                itinerary.get("carrierCode", "Unknown")
            )

            formatted_flight = {
                f"Price ({currency})": float(flight.get("price", {}).get("total", 0)),
                "Duration": itinerary.get("duration", "Unknown")[2:].lower(),
                "Flight Number": f"{itinerary.get('carrierCode', 'XX')} {itinerary.get('number', '000')}",
                "Airline": airline_name,
                "Departure Time": itinerary.get("departure", {}).get("at", "Unknown"),
            }
            flight_list.append(formatted_flight)

        if len(flight_list) == 0:
            return f"No direct flights found between {departure_city}, {departure_country} and {destination_city}, {destination_country} on {travel_date}"

        df = pd.DataFrame(sorted(flight_list, key=lambda x: x[f"Price ({currency})"]))
        return df.to_string(index=False)


# Example Usage:
# if __name__ == "__main__":
# flight_tool = AmadeusFlightSearchTool()
# flights = flight_tool.forward(
#     "London", "Great Britain", "Rome", "Italy", "2025-03-05"
# )
# print(flights)
