import folium
import operator
from Instance import Instance


def create_map(inst):

	loc_data = inst.paxs_longlat + inst.taxis_longlat
	# La instancia tiene longitud / latitud, al reves de lo que se necesita en folium.
	locations = [(l[1],l[0]) for l in loc_data]
	sw = (min(locations, key=operator.itemgetter(0)), min(locations, key=operator.itemgetter(1)))
	ne = (max(locations, key=operator.itemgetter(0)), max(locations, key=operator.itemgetter(1)))

	mymap = folium.Map(location=locations[0])
	mymap.fit_bounds([sw, ne])

	return mymap


def visualize_instance(mymap, inst):
	for lon, lat in inst.taxis_longlat:
		folium.Marker(location=[lat, lon], icon=folium.Icon(color='green')).add_to(mymap)

	for lon, lat in inst.paxs_longlat:
		folium.Marker(location=[lat, lon], icon=folium.Icon(color='red')).add_to(mymap)


def main():
	
	filename = '../../Desktop/Modelos de Decision/PDFs /tp2/input/small_0.csv'
	inst = Instance(filename)

	mymap = create_map(inst)
	# Visualizamos pasajeros y taxis.
	visualize_instance(mymap, inst)

	mymap.save("map.html")


if __name__ == '__main__':
	main()

