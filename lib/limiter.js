const Bottleneck = require('bottleneck');

class RateLimiter {
  constructor({
    maxRequestsPerMinute
  , maxTokensPerMinute
  , bufferPercentage=0.80
  }) {
    this.requestLimiter = new Bottleneck({
      reservoir: Math.floor(maxRequestsPerMinute * bufferPercentage),
      reservoirRefreshAmount: Math.floor(maxRequestsPerMinute * bufferPercentage),
      reservoirRefreshInterval: 60 * 1000,
    });

    this.tokenLimiter = new Bottleneck({
      reservoir: Math.floor(maxTokensPerMinute * bufferPercentage),
      reservoirRefreshAmount: Math.floor(maxTokensPerMinute * bufferPercentage),
      reservoirRefreshInterval: 60 * 1000
    });
  }

  async process(func) {
    return this.requestLimiter.schedule(() =>
      this.tokenLimiter.schedule(async () => {
        let tokensUsed = 0;
        const reportTokens = (count) => { tokensUsed = count; };

        const response = await func(reportTokens);

        await this.tokenLimiter.incrementReservoir(-tokensUsed);

        return response;
      })
    );
  }
}

module.exports = {
  RateLimiter
};