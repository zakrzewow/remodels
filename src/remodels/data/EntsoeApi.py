"""EntsoeApi class"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import datetime as dt

logging.basicConfig(level=logging.INFO)

class EntsoeApi:
    def __init__(self, security_token: str):
        """
        Initializes the EntsoeApi class.

        :param security_token: The security token for accessing the API.
        """
        self.base_url = "https://web-api.tp.entsoe.eu/api"
        self.security_token = security_token
        self.namespace = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
        self.logger = logging.getLogger(__name__)

    def _make_request(self, start_date: dt.date, end_date: dt.date, params: dict, data_parser, resolution_preference=None) -> pd.DataFrame:
        """
        Makes a request to the API and retrieves the data.

        :param start_date: The start date for the data retrieval.
        :param end_date: The end date for the data retrieval.
        :param params: The parameters for the API request.
        :param data_parser: The parser function for the API response.
        :param resolution_preference: Optional resolution preference for the data parser.

        :return: A DataFrame containing the retrieved data.
        :rtype: pd.DataFrame
        """
        data = []
        current_date = start_date

        while current_date < end_date:
            next_date = min(end_date, current_date + dt.timedelta(days=365))
            params.update({
                "periodStart": current_date.strftime("%Y%m%d%H%M"), 
                "periodEnd": next_date.strftime("%Y%m%d%H%M")
            })

            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                # Check if data_parser expects a resolution preference
                if resolution_preference is not None:
                    data += data_parser(response.content, resolution_preference)
                else:
                    data += data_parser(response.content)
                current_date = next_date + dt.timedelta(days=1)
            except requests.HTTPError as e:
                self.logger.error(f"Failed to retrieve data: {e}")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                break

        return pd.DataFrame(data).set_index("datetime") if data else pd.DataFrame()

    def _parse_production_and_load_data(self, content: str) -> list:
        """
        Parses the production and load data from the API's XML response.

        :param content: XML content to parse.
        :type content: str
        :return: List of dictionaries containing production and load data.
        :rtype: list
        """
        root = ET.fromstring(content)
        data = []

        for time_series in root.findall(".//ns:TimeSeries", self.namespace):
            # Parse and calculate the starting interval.
            interval_start = pd.Timestamp(time_series.find(".//ns:Period/ns:timeInterval/ns:start", self.namespace).text)
            # Calculate the resolution in minutes.
            resolution_period = time_series.find(".//ns:Period/ns:resolution", self.namespace).text
            resolution_minutes = int(resolution_period[2:-1])  # Expects format 'PT15M' or 'PT60M'.
            resolution = dt.timedelta(minutes=resolution_minutes)

            # Extract the PSR type.
            psr_type_elem = time_series.find(".//ns:psrType", self.namespace)
            psr_type = psr_type_elem.text if psr_type_elem is not None else None

            # Process each point in the time series.
            for point in time_series.findall(".//ns:Point", self.namespace):
                position = int(point.find("ns:position", self.namespace).text)
                datetime_position = interval_start + (resolution * (position - 1))
                quantity = float(point.find("ns:quantity", self.namespace).text)

                data_point = {"datetime": datetime_position, "quantity": quantity}
                if psr_type:
                    data_point["PSRType"] = psr_type
                data.append(data_point)

        return data

    def _parse_pricing_data(self, content: str, resolution_preference: int = None) -> list:
        """
        Parses the pricing data from the API's XML response.

        :param content: XML content to parse.
        :type content: str
        :param resolution_preference: The preferred data resolution in minutes, if applicable.
        :type resolution_preference: int, optional
        :return: List of dictionaries containing pricing data.
        :rtype: list
        """
        root = ET.fromstring(content)
        data = []
        
        # Extract and process data for each period.
        for period in root.findall('.//ns:Period', namespaces=self.namespace):
            resolution_text = period.find('ns:resolution', namespaces=self.namespace).text
            resolution_minutes = int(resolution_text[2:-1])  # Expects format 'PT15M' or 'PT60M'.
            resolution_timedelta = dt.timedelta(minutes=resolution_minutes)
            
            # Skip entries that don't match the desired resolution.
            if resolution_preference and resolution_minutes != resolution_preference:
                continue
            
            time_interval = period.find('ns:timeInterval', namespaces=self.namespace)
            start_time = pd.Timestamp(time_interval.find('ns:start', namespaces=self.namespace).text)
            
            for point in period.findall('ns:Point', namespaces=self.namespace):
                position = int(point.find('ns:position', namespaces=self.namespace).text)
                price_amount = float(point.find('ns:price.amount', namespaces=self.namespace).text)
                exact_time = start_time + resolution_timedelta * (position - 1)
                data.append({
                    'price_da': price_amount, 
                    'datetime': exact_time
                })

        return data

    def get_day_ahead_pricing(self, start_date: dt, end_date: dt, in_domain: str, resolution_preference: int = None) -> pd.DataFrame:
        """
        Retrieves day-ahead pricing data from the API for a given domain and date range.

        :param start_date: The start date for the data retrieval.
        :type start_date: datetime
        :param end_date: The end date for the data retrieval.
        :type end_date: datetime
        :param in_domain: The market domain for which to retrieve pricing data.
        :type in_domain: str
        :param resolution_preference: The resolution in minutes for the pricing data (optional).
        :type resolution_preference: int, optional
        :return: A DataFrame containing day-ahead pricing data.
        :rtype: pd.DataFrame
        """
        # Define the parameters for the API request
        params = {
            "securityToken": self.security_token,
            "documentType": "A44",
            "ProcessType": "A33",
            "in_Domain": in_domain,
            "out_Domain": in_domain
        }
        # Make the request and return the parsed data as a DataFrame
        return self._make_request(start_date, end_date, params, self._parse_pricing_data, resolution_preference)

    def get_forecast_load(self, start_date: dt, end_date: dt, out_domain: str) -> pd.DataFrame:
        """
        Retrieves forecasted load data from the API for a given domain and date range.

        :param start_date: The start date for the data retrieval.
        :type start_date: datetime
        :param end_date: The end date for the data retrieval.
        :type end_date: datetime
        :param out_domain: The market domain for which to retrieve load forecast data.
        :type out_domain: str
        :return: A DataFrame containing forecasted load data.
        :rtype: pd.DataFrame
        """
        # Define the parameters for the API request
        params = {
            "securityToken": self.security_token,
            "documentType": "A65",
            "processType": "A01",
            "OutBiddingZone_Domain": out_domain
        }
        # Make the request and return the parsed data as a DataFrame
        return self._make_request(start_date, end_date, params, self._parse_production_and_load_data)
