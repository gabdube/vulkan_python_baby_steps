from enum import Enum
from collections import namedtuple


class EventsMap(dict):

    def __iter__(self):
        events = tuple(self.keys())
        for e in events:
            data = self[e]

            if e in events_to_replace:
                del self[e]
            
            yield e, data

    def __setitem__(self, event, event_data):
        if event in events_to_replace:
            super().__setitem__(event, event_data)
        else:
            raise KeyError(f"Invalid event type: {key}")


Events = Enum("Events", "WindowResized")

WindowResized = Events.WindowResized
WindowResizedData = namedtuple('WindowResizedData', 'width height')

events_to_replace = (WindowResized,)
