import geopandas as gpd
import shapely
from pyproj import Geod
from shapely import wkt


class GeodataFrameFilter:
    ''' Basic Filtering of GeodataFrame '''

    def __init__(self, data, area=0, multipolygons=False):
        self.data = data
        self.area = area
        self.multipolygons = multipolygons

    def onlyPoly(self):
        """
        Get only Polygons from geodata frame
        """
        newgpd = gpd.GeoDataFrame(columns=self.data.columns)
        for idx, row in self.data.iterrows():
            if (isinstance(row.geometry, shapely.geometry.polygon.Polygon)):
                newgpd = newgpd.append(row, ignore_index=True)
        self.data = newgpd

    def get_area(self, polygon):
        """
        get area as sqm
        """
        polygon = str(polygon)
        geod = Geod(ellps="WGS84")
        return abs(geod.geometry_area_perimeter(wkt.loads(polygon))[0])

    def filter(self):

        if self.multipolygons:
            self.onlyPoly()
            self.data['area'] = self.data['geometry'].map(
                lambda x: self.get_area(x))
            self.data = self.data[self.data['area'] > self.area]
            return self.data
        else:
            self.data['area'] = self.data['geometry'].map(
                lambda x: self.get_area(x))
            self.data = self.data[self.data['area'] > self.area]
            return self.data
