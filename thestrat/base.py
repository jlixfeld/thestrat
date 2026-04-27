"""DEFUNCT — placeholder for the pre-1.0 thestrat.base module.

`thestrat` was rewritten in 1.0.0. This module no longer exists.
Importing it indicates the importing project pins a defunct version of
this package and needs to be deleted or migrated.
"""

raise ImportError(
    "thestrat has been rewritten in 1.0.0. The module 'thestrat.base' "
    "no longer exists.\n"
    "If you hit this, the project that imported it is using a defunct "
    "version of thestrat — delete the project or migrate it to the 1.0 API:\n"
    "    from thestrat import TimeframeAggregator, classify_bars_df"
)
