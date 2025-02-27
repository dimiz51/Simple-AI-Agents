import os
import requests
from dotenv import load_dotenv
from smolagents import Tool
from typing import List
import pandas as pd
import re


class AmadeusHotelFinderTool(Tool):
    name = "amadeus_hotel_finder"
    description = "Finds available hotel rooms, in a given city/country and look for the best offers. Should be used as default for finding hotels."
    inputs = {
        "city_name": {
            "type": "string",
            "description": "Name of the city to search hotels in. Example: 'New York'.",
        },
        "radius": {
            "type": "integer",
            "description": "Radius of the search in kilometers. Default can be 15 km.",
        },
        "country": {
            "type": "string",
            "description": "Name of the country to search hotels in. Example: 'United States of America'.",
        },
        "num_adults": {
            "type": "integer",
            "description": "Number of adults in the room.",
        },
        "check_in_date": {
            "type": "string",
            "description": "Check-in date in YYYY-MM-DD format.",
        },
        "stay_days": {"type": "integer", "description": "Number of days to stay."},
        "price_range": {
            "type": "string",
            "description": "Optional price range in the format 'min-max'.",
            "nullable": True,
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

    def get_access_token(self) -> str:
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

    def get_city_code(self, city_name: str, token: str, country_code: str) -> str:
        """Retrieves the IATA city code for a given city name."""
        url = f"{self.base_url}/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            url, params={"keyword": city_name, "subType": "CITY"}, headers=headers
        )
        response.raise_for_status()

        city_data = response.json().get("data", [])
        if country_code:
            city_data = [
                city
                for city in city_data
                if city.get("address", {}).get("countryCode") == country_code
            ]

        if not city_data:
            raise ValueError(
                f"Could not find city code for {city_name}. Maybe it's not available."
            )

        return city_data[0]["iataCode"]

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

    def fetch_hotels(self, city_code: str, token: str, radius: int) -> List[str]:
        """Fetches available hotel IDs in a given city using the city code."""
        url = f"{self.base_url}/v1/reference-data/locations/hotels/by-city"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            url,
            params={
                "cityCode": city_code,
                "radius": radius,
                "radiusUnit": "KM",
                "hotelSource": "ALL",
            },
            headers=headers,
        )
        response.raise_for_status()

        hotels_data = response.json().get("data", [])
        if not hotels_data:
            raise ValueError(f"No hotels found for city with code {city_code}")

        return [hotel["hotelId"] for hotel in hotels_data]

    def fetch_hotel_offers(
        self,
        hotel_ids: List[str],
        check_in: str,
        check_out: str,
        adults: int,
        price_range: str,
        currency: str,
        token: str,
    ) -> str:
        """Fetches hotel offers and returns them as a formatted DataFrame string."""
        url = f"{self.base_url}/v3/shopping/hotel-offers"
        data = []

        for hotel_id in hotel_ids[:50]:
            try:
                params = {
                    "hotelIds": str(hotel_id),
                    "adults": adults,
                    "checkInDate": check_in,
                    "checkOutDate": check_out,
                    "priceRange": price_range if price_range else None,
                    "currency": currency if currency else None,
                    "bestRateOnly": "true",
                }

                headers = {"Authorization": f"Bearer {token}"}

                response = requests.get(
                    url,
                    params={k: v for k, v in params.items() if v is not None},
                    headers=headers,
                )

                if response.status_code != 200:
                    raise ValueError(
                        f"Failed to fetch hotel offers: {response.json().get('errors', [{}])[0].get('detail', response.json())}"
                    )

                offers = response.json().get("data", [])
                if not offers:
                    return "No hotel offers found for this hotel."
                for offer in offers:
                    hotel_name = offer.get("hotel", {}).get("name", "Unknown Hotel")
                    latitude = offer.get("hotel", {}).get("latitude", "N/A")
                    longitude = offer.get("hotel", {}).get("longitude", "N/A")
                    price = (
                        offer["offers"][0]["price"]["total"]
                        if "offers" in offer
                        else "N/A"
                    )

                    data.append(
                        {
                            "Hotel Name": hotel_name,
                            "Price": f"{price} {currency}",
                            "Check-in Date": check_in,
                            "Check-out Date": check_out,
                            "Latitude": latitude,
                            "Longitude": longitude,
                        }
                    )
            except Exception as e:
                continue

            # We don't need to overload the API for this demo...
            if len(data) >= 15:
                break

        df = pd.DataFrame(data)
        return df.to_string(index=False)

    def forward(
        self,
        city_name: str,
        radius: int,
        country: str,
        num_adults: int,
        check_in_date: str,
        stay_days: int,
        price_range: str = None,
        currency: str = None,
    ) -> str:
        """Main method that retrieves the best hotel offers in a given city."""
        # Check format for price range and check-in date
        if price_range and not re.match(r"\d+-\d+", price_range):
            raise ValueError("Invalid price range format. Please use min-max format.")

        if not re.match(r"\d{4}-\d{2}-\d{2}", check_in_date):
            raise ValueError(
                "Invalid date format. Please use YYYY-MM-DD format for the check-in date."
            )

        token = self.get_access_token()
        country_code = self.convert_country_to_code(country)
        city_code = self.get_city_code(city_name, token, country_code)
        hotel_ids = self.fetch_hotels(city_code, token, radius)

        if not hotel_ids:
            raise ValueError(f"No hotels found in {city_name}, {country}")

        currency = currency if currency else "USD"

        check_out_date = (
            pd.to_datetime(check_in_date) + pd.Timedelta(days=stay_days)
        ).strftime("%Y-%m-%d")
        hotel_offers = self.fetch_hotel_offers(
            hotel_ids,
            check_in_date,
            check_out_date,
            num_adults,
            price_range,
            currency,
            token,
        )

        if hotel_offers is None:
            raise ValueError(f"No hotel offers found in {city_name}, {country}")

        return hotel_offers


# Example usage
# if __name__ == "__main__":
#     hotel_tool = AmadeusHotelFinderTool()
#     hotels = hotel_tool.forward("London", 20, "Great Britain", 1, "2025-03-15", 1)
#     print(hotels)
