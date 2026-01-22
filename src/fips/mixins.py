class AttributeMapperMixin:
    """
    A mixin that automatically generates properties to map
    subclass attribute names to base class attribute names.
    """

    _attribute_map = {}
    _read_only_attributes = set()

    # We use __init_subclass__ to generate properties when the class is DEFINED,
    # rather than catching access when the code runs.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for alias, target in cls._attribute_map.items():
            cls._create_alias_property(alias, target)

    @classmethod
    def _create_alias_property(cls, alias, target):
        """Helper to create and attach the property to the class."""

        # Define the Getter
        def getter(self):
            return getattr(self, target)

        # Define the Setter
        if alias in cls._read_only_attributes:
            setter = None  # Makes it read-only
        else:

            def setter(self, value):
                setattr(self, target, value)

        # Attach the property to the class
        setattr(cls, alias, property(getter, setter))
