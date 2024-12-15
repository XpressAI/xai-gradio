from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, SubGraphExecutor

import requests
from datetime import datetime, timezone

@xai_component
class GetWeather(Component):
    """Fetches current weather information for a specified city.

    ##### inPorts:
    - city (str): The name of the city to retrieve weather for.
    - api_key (str): The API key for authentication.
    - url (str): The base URL of the weather API.

    ##### outPorts:
    - weather_info (str): The weather information as a JSON string.
    """
    city: InArg[str]
    api_key: InArg[str]
    url: InArg[str]
    weather_info: OutArg[str]

    def execute(self, ctx) -> None:
        try:
            city_name = self.city.value
            api_key = self.api_key.value
            base_url = self.url.value

            if not city_name:
                raise ValueError("City name is required.")
            if not api_key:
                raise ValueError("API key is required.")
            if not base_url:
                raise ValueError("API base URL is required.")

            url = f"{base_url}?q={city_name}&appid={api_key}&units=metric&lang=en"

            response = requests.get(url)
            response.raise_for_status()  

            data = response.json()
            weather_description = data['weather'][0]['description']
            temperature = data['main']['temp']
            humidity = data['main']['humidity']

            weather_info = {
                "city": city_name,
                "description": weather_description,
                "temperature": temperature,
                "humidity": humidity,
            }

            self.weather_info.value = str(weather_info)

        except requests.RequestException as e:
            self.weather_info.value = f"Error fetching data: {str(e)}"
        except KeyError as e:
            self.weather_info.value = f"Missing key in the received data: {str(e)}"
        except ValueError as e:
            self.weather_info.value = f"Input error: {str(e)}"
