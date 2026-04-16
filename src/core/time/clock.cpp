#include "core/time/clock.hpp"
#include <algorithm>

namespace LX_core {

void Clock::tick() {
  const Clk::time_point now = Clk::now();
  if (m_firstTick) {
    m_startTime = now;
    m_lastTickTime = now;
    m_deltaTime = 0.0f;
    m_totalTime = 0.0;
    m_frameCount = 0;
    m_firstTick = false;
    return;
  }

  m_deltaTime = std::chrono::duration<float>(now - m_lastTickTime).count();
  m_totalTime = std::chrono::duration<double>(now - m_startTime).count();
  m_lastTickTime = now;
  ++m_frameCount;

  m_recentDeltas[m_smoothCursor] = m_deltaTime;
  m_smoothCursor = (m_smoothCursor + 1) % kSmoothWindow;
  if (m_sampleCount < kSmoothWindow) {
    ++m_sampleCount;
  }
}

float Clock::smoothedDeltaTime() const {
  if (m_sampleCount == 0) {
    return m_deltaTime;
  }
  float sum = 0.0f;
  for (int i = 0; i < m_sampleCount; ++i) {
    sum += m_recentDeltas[i];
  }
  return sum / static_cast<float>(m_sampleCount);
}

} // namespace LX_core
