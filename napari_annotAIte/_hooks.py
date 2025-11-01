"""
Napari plugin hooks for discovery and registration.
"""


def napari_experimental_provide_dock_widget():
    """
    Napari hook that returns our widget factory.
    
    This function is called by napari to discover available dock widgets.
    Returns a list of widget factories (functions or magicgui-decorated functions).
    """
    from ._widget import falsecolor_widget
    return [falsecolor_widget]

