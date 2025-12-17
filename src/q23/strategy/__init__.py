"""q23.strategy

Strategy package (Quantiacs implementation).

Modules are organized to keep boundaries clean:
- data_loader: data access / universe selection (qnt.data)
- factors: factor library construction
- ic_weighting: dynamic IC weighting & composite score
- portfolio: portfolio construction / constraints / transaction costs
- outputs: dashboard-ready CSV exports
- engine: orchestration (ties everything together)
"""
