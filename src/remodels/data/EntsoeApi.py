import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import datetime as dt

logging.basicConfig(level=logging.INFO)

PSRTYPE_DICT = {"A03":"Mixed",
            "A04":"Generation",
            "A05": "Load",
            "B01": "Biomass",
            "B02": "Fossil Brown coal/Lignite",
            "B03": "Fossil Coal-derived gas",
            "B04": "Fossil Gas",
            "B05": "Fossil Hard coal",
            "B06": "Fossil Oil",
            "B07": "Fossil Oil shale",
            "B08": "Fossil Peat",
            "B09": "Geothermal",
            "B10": "Hydro Pumped Storage",
            "B11": "Hydro Run-of-river and poundage",
            "B12": "Hydro Water Reservoir",
            "B13": "Marine",
            "B14": "Nuclear",
            "B15": "Other renewable",
            "B16": "Solar",
            "B17": "Waste",
            "B18": "Wind Offshore",
            "B19": "Wind Onshore",
            "B20": "Other",
            "B21": "AC Link",
            "B22": "DC Link",
            "B23": "Substation",
            "B24": "Transformer"}

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

    def _make_request(self, start_date: dt.date, end_date: dt.date, params: dict, data_parser) -> pd.DataFrame:
        """
        Makes a request to the API and retrieves the data.

        :param start_date: The start date for the data retrieval.
        :param end_date: The end date for the data retrieval.
        :param params: The parameters for the API request.
        :param data_parser: The parser function for the API response.

        :return: A DataFrame containing the retrieved data.
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
                data += data_parser(response.content)
                current_date = next_date + dt.timedelta(days=1)
            except requests.HTTPError as e:
                self.logger.error(f"Failed to retrieve data: {e}")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                break

        return pd.DataFrame(data).set_index("datetime") if data else pd.DataFrame()

    def _parse_production_and_load_data(self, content):
        root = ET.fromstring(content)
        data = []

        for time_series in root.findall(".//ns:TimeSeries", self.namespace):
            interval_start = pd.Timestamp(
                pd.Timestamp(time_series.find(".//ns:Period/ns:timeInterval/ns:start", self.namespace).text)
            )

            resolution = dt.timedelta(
                minutes=int(time_series.find(".//ns:Period/ns:resolution", self.namespace).text[2:-1])
            )

            psr_type_elem = time_series.find(".//ns:psrType", self.namespace)
            psr_type = psr_type_elem.text if psr_type_elem is not None else None

            for point in time_series.findall(".//ns:Point", self.namespace):
                position = int(point.find("ns:position", self.namespace).text)
                datetime_position = interval_start + pd.Timedelta(hours=position-1)  # Modified this line
                quantity = float(point.find("ns:quantity", self.namespace).text)

                data_point = {"datetime": datetime_position, "quantity": quantity}
                if psr_type is not None:
                    data_point["PSRType"] = psr_type

                data.append(data_point)
        return data

    def _parse_pricing_data(self, content):
        root = ET.fromstring(content)
        data = []
        
        # Extract data
        data = []
        namespace = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'}
        for period in root.findall('.//ns:Period', namespaces=namespace):
            time_interval = period.find('ns:timeInterval', namespaces=namespace)
            start_time = pd.Timestamp(time_interval.find('ns:start', namespaces=namespace).text)

            for point in period.findall('ns:Point', namespaces=namespace):
                position = int(point.find('ns:position', namespaces=namespace).text)
                price_amount = float(point.find('ns:price.amount', namespaces=namespace).text)
                exact_time = start_time + dt.timedelta(hours=position - 1)
                data.append({
                    'price_da': price_amount, 
                    'datetime': exact_time
                })

        return data

    def get_actual_production(self, start_date, end_date, in_domain):
        params = {
            "securityToken": self.security_token,
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": in_domain
        }
        actual_production = self._make_request(start_date, end_date, params, self._parse_production_and_load_data)
        actual_production["PSRType"]=actual_production["PSRType"].map(PSRTYPE_DICT)
        return actual_production

    def get_actual_load(self, start_date, end_date, out_domain):
        params = {
            "securityToken": self.security_token,
            "documentType": "A65",
            "processType": "A16",
            "OutBiddingZone_Domain": out_domain
        }
        return self._make_request(start_date, end_date, params, self._parse_production_and_load_data)

    def get_day_ahead_pricing(self, start_date, end_date, in_domain):
        params = {
            "securityToken": self.security_token,
            "documentType": "A44",
            "ProcessType": "A33",
            "in_Domain": in_domain,
            "out_Domain": in_domain
        }
        return self._make_request(start_date, end_date, params, self._parse_pricing_data)
    
    def get_forecast_load(self, start_date, end_date, out_domain):
        params = {
            "securityToken": self.security_token,
            "documentType": "A65",
            "processType": "A01",
            "OutBiddingZone_Domain": out_domain
        }
        return self._make_request(start_date, end_date, params, self._parse_production_and_load_data)

    def get_data(self, start_date: dt.date, end_date: dt.date, country_codes: dict) -> pd.DataFrame:
        """
        Retrieves data for the specified countries and date range.

        :param start_date: The start date for data retrieval.
        :param end_date: The end date for data retrieval.
        :param country_codes: A dictionary containing country codes and their respective domains.

        :return: A DataFrame containing the retrieved data.
        """
        def download_data(method, start_date, end_date):
            data_frames = {}
            for country_code, domain in country_codes.items():
                data = method(start_date, end_date, domain)
                if not data.empty:
                    data["country"] = country_code
                    data_frames[country_code] = data
                else:
                    self.logger.warning(f"No data available for {country_code}")

            return pd.concat(data_frames.values()) if data_frames else pd.DataFrame()

        self.logger.info("Downloading price data")
        prices = download_data(self.get_day_ahead_pricing, start_date, end_date)

        self.logger.info("Downloading load data")
        load = download_data(self.get_forecast_load, start_date, end_date)

        if not prices.empty and not load.empty:
            merged_data = prices.merge(load, on=["datetime", "country"])
            merged_data = merged_data.rename({"quantity": "load"}, axis=1)
            return merged_data
        else:
            self.logger.warning("Data is not available, returning an empty DataFrame")
            return pd.DataFrame()



    